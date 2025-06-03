# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os

import torch
import numpy as np
from urdfpy import URDF
import yaml

from fabrics_sim.fabrics.model_batch_builder import create_model_batch
from fabrics_sim.fabrics.taskmap_container import TaskmapContainer
from fabrics_sim.utils.utils import jvp, jacobian
from fabrics_sim.utils.path_utils import get_robot_urdf_path, get_params_path 
from fabrics_sim.utils.math_utils import inverse_pd_matrix

import warp as wp
import warp.sim
from warp.sim.import_urdf import parse_urdf

import time

# NOTE: Got this from Nathan.
@wp.kernel
def accel_constraint_proj_kernel(
        # inputs
        batch_qdd: wp.array(dtype=float, ndim=2),
        cspace_dim: int,
        accel_limits: wp.array(dtype=float, ndim=1),
        # output (can be the same as the corresponding batch_qdd input)
        batch_qdd_scaled: wp.array(dtype=float, ndim=2)):
    batch_index = wp.tid()
    min_scalar = float(1.)

    for i in range(cspace_dim):
        qdd_abs = abs(batch_qdd[batch_index, i])
        accel_limit = accel_limits[i]

        if qdd_abs > accel_limit:
            scalar = accel_limit / qdd_abs
            if scalar < min_scalar:
                min_scalar = scalar

    for i in range(cspace_dim):
        batch_qdd_scaled[batch_index, i] = min_scalar * batch_qdd[batch_index, i]

class AccelConstraint(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, qdd, allocated_data):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and ouputs
        ctx.qdd = wp.torch.from_torch(qdd)
        ctx.allocated_data = allocated_data

        with ctx.tape:
            batch_size = qdd.shape[0]
            wp.launch(kernel=accel_constraint_proj_kernel,
                      dim=batch_size,
                      inputs=[
                          ctx.qdd,
                          ctx.allocated_data['cspace_dim'],
                          ctx.allocated_data['accel_limits']],
                      outputs=[
                          ctx.allocated_data['qdd_scaled']],
                      device=ctx.allocated_data['device'])

        return wp.torch.to_torch(ctx.allocated_data['qdd_scaled'])

    @staticmethod
    def backward(ctx, adj_qdd_scaled):
        #grads = { ctx.allocated_data['qdd_scaled']:
        #            wp.torch.from_torch(adj_qdd_scaled) }

        ctx.allocated_data['qdd_scaled'].grad = wp.torch.from_torch(adj_qdd_scaled)
    
        # Calculate gradients
        #ctx.tape.backward(grads=grads)
        ctx.tape.backward()

        # Return adjoint w.r.t. inputs
        #return (wp.torch.to_torch(ctx.tape.gradients[ctx.qdd]), None)
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.qdd]), None)

class BaseFabric(torch.nn.Module):
    """
    Implements base graph for combining fabric components together.
    Handles the creation of task map containers which hold task maps,
    fabric terms, and energies. Performs pullback and combine operations.
    """
    def __init__(self, device, batch_size, timestep, fabric_params_filename, fabric_params=None,
                 graph_capturable=True):
        # TODO: need to place comments on inputs here
        """
        Constructor.
        -----------------------------------------
        @param device: type str that sets the cuda device for the fabric
        @param fabric_params_filename: str, filename for the fabric parameters
        @param fabric_params: dict, of fabric parameters
        """
        self.batch_size = batch_size
        self.graph_capturable = graph_capturable
        super(BaseFabric, self).__init__()

        # Preload all kernels to get around issues where at times a kernel can't be found. Happens
        # With multi-gpu training for some reason.
        #wp.force_load(device)

        # This is a list of containers that hold task maps with associated fabric and energy terms.
        # One container holds one task map, but potentially many fabric and energy terms.
        self.taskmap_containers = dict()
        # Dictionary of features that are passed to fabric terms.
        self.fabrics_features = dict()
        # Dictionary of external perturbation force associated with a taskmap.
        self.external_forces = dict()
        # Robot model
        self.model = None
        self.robot_name = None
        self.urdfpy_robot = None
        self.batch_size = batch_size
        # List of NN modules
        self.module_list = torch.nn.ModuleList()

        # Save which device the fabric will live on
        self.device = device

        # Fabric parameters
        self.fabric_params = None
        if fabric_params is not None:
            self.fabric_params = fabric_params
        else:
            self.load_params(fabric_params_filename)

        # Extra data for acceleration and jerk limiting.
        # TODO: use jerk limits, accel limits, and jerk model to back out new acceleration limits
        # that also respect jerk limits.
        # TODO: need to make this update if there is a shift in timestep
        self.accel_limits = torch.tensor(self.fabric_params['joint_limits']['acceleration'], device=self.device)
        self.jerk_limits = torch.tensor(self.fabric_params['joint_limits']['jerk'], device=self.device)

        # Check to make sure number of values specific for jerk and accel constraints are the same
        assert self.accel_limits.shape[0] == self.jerk_limits.shape[0],\
            "Number of values for accel limits must match that of jerk limits in yaml."

        # Allocate data needed for acceleration and jerk limiting.
        self.allocated_data = { 'cspace_dim': self.accel_limits.shape[0],
                                'accel_limits': None,
                                'qdd_scaled': None,
                                'timestep': None,
                                'device': self.device }

        self.update_accel_limits(timestep)

        self.joint_names = []

    def update_accel_limits(self, timestep):
        """
        Calculate new acceleration limits that respect both the original acceleration limits
        and jerk limits.
        -----------------------------------------
        @param timestep: timestep for fabric
        """
        # If there is a change in timestep, we need to update the acceleration limit.
        if self.allocated_data['timestep'] is None or\
                abs(self.allocated_data['timestep'] - timestep) > 1e-6:
            # Set new acceleration limits to original acceleration limits.
            new_accel_limits = torch.clone(self.accel_limits).detach()
            for i in range(new_accel_limits.shape[0]):
                # If acceleration limit is too aggressive for jerk limit, then
                # update acceleration limit.
                if (2. * self.accel_limits[i]) / timestep > self.jerk_limits[i]:
                    new_accel_limits[i] = (self.jerk_limits[i] * timestep) / 2.

            # Save updated accel limits
            self.allocated_data['accel_limits'] =\
                    wp.torch.from_torch(new_accel_limits)
            
            # Save updated timestep
            self.allocated_data['timestep'] = timestep

    def load_params(self, fabric_params_filename):
        path = get_params_path()
        config_path = os.path.join(path, fabric_params_filename)
        with open(config_path, 'r') as file:
            self.fabric_params = yaml.safe_load(file)

        self.fabric_params = self.fabric_params['fabric_params']

    def add_taskmap(self, taskmap_name, taskmap, graph_capturable):
        """
        Creates a taskmap container by passing the name of and task map itself.
        -----------------------------
        @param taskmap_name: name for the task map
        @param taskmap: taskmap itself
        """
        # Create new taskmap container with taskmap name if it doesn't already exist and
        # add to containers.
        if taskmap_name not in self.taskmap_containers:
            self.taskmap_containers[taskmap_name] =\
                TaskmapContainer(taskmap_name, taskmap, graph_capturable=graph_capturable)

        # Set fabric features associated with this task map to None if features do not yet
        # exist.
        # Set external forces assicated with this task map to None to initialize.
        if taskmap_name not in self.fabrics_features:
            self.fabrics_features[taskmap_name] = None
            self.external_forces[taskmap_name] = None
    
    def add_fabric(self, taskmap_name, fabric_name, fabric):
        """
        Adds a fabric term to an existing taskmap container.
        -----------------------------
        @param taskmap_name: name for the task map
        @param fabric_name: name for the fabric term
        @param fabric: fabric term itself
        """
        # Find target taskmap container and add fabric to this container.
        self.taskmap_containers[taskmap_name].add_fabric(fabric_name, fabric)

        # Allocate features and associate with both taskmap and fabric.
        try:
            self.fabrics_features[taskmap_name][fabric_name] = None
        except:
            if self.fabrics_features[taskmap_name] is None:
                self.fabrics_features[taskmap_name] = { fabric_name: None }
            else:
                self.fabrics_features[taskmap_name][fabric_name] = None

    def add_energy(self, taskmap_name, energy_name, energy):
        """
        Adds an energy term to an existing taskmap container.
        -----------------------------
        @param taskmap_name: name for the task map
        @param energy_name: name for the energy term
        @param energy: energy term itself
        """
        # Find target taskmap container and add energy to this container.
        self.taskmap_containers[taskmap_name].add_energy(energy_name, energy)

    def eval_container(self, container, q, qd, fabric_features, external_force):
        """
        Evaluates a taskmap container by evaluating the task map states,
        evaluating the fabric and energy terms in this task map, and pulling
        back and combining all components from the task map.
        -----------------------------
        @param container: taskmap container itself
        @param q: root position
        @param qd: root velocity
        @param fabric_features: dictionary of features associated with components
        living in this task map
        @return root_metric: metric(mass) of the combined system in the root
        @return root_geometric_force: force produced by geometries combined in the root
        @return root_potential_force: force produced by potentials combined in the root
        @return root_energy_metric: metric(mass) of the energy in the root
        @return root_energy_force: force produced by energy combined in the root
        @return energy: energy from this task map
        """
        
        x = None
        jac = None
        curvature_force = None
        xd = None
        # If jacobian is None, then we will build jacobian via auto diff
        # and use double back to obtain taskmap velocity and curvature force.
        # It is assumed that the forward map is constructed purely from PyTorch.
        if False: #jac is None:
            # TODO: this path is broken. Currently the following functions
            # don't work correctly while maintaining gradient functions.
            #jac = jacobian(x, q)
            #xd = jvp(x, q, qd, True, True)
            #curvature_force = jvp(xd, q, qd, True, True)
            raise('Must provide a Jacobian return')
        # If Jacobian was calculated, then do typical method for calculating
        # taskspace velocity and curvature force.
        else:
            # First calculate forward pass at perturbed root position
            # because we have functions that return the last taskspace
            # position and Jacobian evals. Therefore, we want to
            # eval the taskspace at the actual root position last so
            # when these taskspace query functions are used, they return
            # data from the actual root position

            # Eval at perturbed position
            eps = 1e-5
            q_eps = q + eps * qd
            x_eps, jac_eps = container.eval_taskmap(q_eps)

            # Eval at actual position
            x, jac = container.eval_taskmap(q)

            # Calculation of curvature force.
            jac_dot = (1./eps) * (jac_eps - jac)
            curvature_force = torch.bmm(jac_dot, qd.unsqueeze(2)).squeeze(2)

            # Eval actual velocity
            xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        
        # TODO: not sure if we still need these.
        q.grad = None
        qd.grad = None

        # Calculate leaf metrics and accelerations.
        M_leaf, potential_force, geometric_force = \
                container.eval_fabrics(x, xd, fabric_features, external_force)

        # Calculate root metric in batch.
        # NOTE: the root_metric is not differentiable because it leverages the jacobian
        # which based on it being constructed via derivates, loses it's differentiability.
        # This is only the case where the forward maps come from Warp because Warp
        # does not yet currently support double derivatives.
        root_metric = None
        if M_leaf is not None:
            root_metric = torch.bmm(torch.bmm(jac.transpose(1,2), M_leaf), jac)

        # Calculate root force in batch.
        # NOTE: placing force on the left side
        #leaf_potential_force = torch.bmm(M_leaf, -xdd_potential.unsqueeze(2)+curvature_force.unsqueeze(2))
        #leaf_geometric_force = torch.bmm(M_leaf, -xdd_geometric.unsqueeze(2)+curvature_force.unsqueeze(2))
        root_potential_force = None
        root_geometric_force = None
        root_curv_force_from_potential = None
        if potential_force is not None:
            leaf_potential_force = potential_force.unsqueeze(2) # + torch.bmm(M_leaf, curvature_force.unsqueeze(2))
            root_potential_force = (torch.bmm(jac.transpose(1,2), leaf_potential_force)).squeeze(2)
            # Need to add the curvature force from the potential force space to the geometric force
            leaf_curv_force = torch.bmm(M_leaf, curvature_force.unsqueeze(2))
            root_curv_force_from_potential = (torch.bmm(jac.transpose(1,2), leaf_curv_force)).squeeze(2)
        if geometric_force is not None:
            leaf_geometric_force = geometric_force.unsqueeze(2) + torch.bmm(M_leaf, curvature_force.unsqueeze(2))
            root_geometric_force = (torch.bmm(jac.transpose(1,2), leaf_geometric_force)).squeeze(2)
        # Need to add the curvature force from the potential force space to the geometric force
        if root_curv_force_from_potential is not None:
            if root_geometric_force is None:
                root_geometric_force = root_curv_force_from_potential
            else:
                root_geometric_force = root_geometric_force + root_curv_force_from_potential

        # Pullback energy mass and force--------------------------------------------------------
        M_energy, energy_force, energy = \
                container.eval_energies(x, xd)

        root_energy_metric = None
        root_energy_force = None
        if M_energy is not None:
            root_energy_metric = torch.bmm(torch.bmm(jac.transpose(1,2), M_energy), jac)
        
            # Force on left side
            leaf_energy_force =\
                    energy_force.unsqueeze(2) + torch.bmm(M_energy, curvature_force.unsqueeze(2))
            root_energy_force = (torch.bmm(jac.transpose(1,2), leaf_energy_force)).squeeze(2)

        return (root_metric, root_geometric_force, root_potential_force,
                root_energy_metric, root_energy_force, energy)

    def eval_natural(self, q, qd, timestep): #, metric_out, force_out, metric_inv_out):
        """
        Calculates the total system metric (mass), M, and force, f, for this equation:
        M qdd + f = 0
        Calculations include system mass, geometric force, potential force, energy mass and force,
        energization, and additional damping.
        -----------------------------
        @param q: batched root position, bxn
        @param qd: batched root velocity, bxn
        @return metric: batched metric (mass) of the resulting fabric, bxnxn
        @return force: batched force of teh resulting fabric, bxn
        @return metric_inv: inverse of batched metric (mass), bxnxn
        """

#        # Check to see if batch size has changed and this fabric is using a robot model
#        if q.shape[0] != self.batch_size and self.model is not None:
#            self.batch_size = q.shape[0]
#            self.load_robot(self.robot_name, self.batch_size)

        # Lists of various components aggregated across task spaces. We append the components
        # to these lists and then collapse the list, by summing across them.
        #root_metrics = []
        #root_geometric_forces = []
        #root_potential_forces = []
        #root_energy_metrics = []
        #root_energy_forces = []
        #energies = []

        # Zero out tensors
        if self.graph_capturable:
            self.root_metrics.zero_().detach_()
            self.root_geometric_forces.zero_().detach_()
            self.root_potential_forces.zero_().detach_()
            self.root_energy_metrics.zero_().detach_()
            self.root_energy_forces.zero_().detach_()
            self.energies.zero_().detach_()
        else:
            self.root_metrics = torch.zeros_like(self.root_metrics)
            self.root_geometric_forces = torch.zeros_like(self.root_geometric_forces)
            self.root_potential_forces = torch.zeros_like(self.root_potential_forces)
            self.root_energy_metrics = torch.zeros_like(self.root_energy_metrics)
            self.root_energy_forces = torch.zeros_like(self.root_energy_forces)
            self.energies = torch.zeros_like(self.energies)

        # Cycle through each taskmap container, evaluating all components in that container, pulling
        # them to the root, and adding the results to the respective lists above.
        for (taskmap_name, container) in self.taskmap_containers.items():
            # Evaluate the taskmap container with results already pulled to the root.
            (root_metric, root_geometric_force, root_potential_force,
                    root_energy_metric, root_energy_force, energy) =\
                            self.eval_container(container, q, qd,
                                                self.fabrics_features[container.name],
                                                self.external_forces[container.name])
            
            # If a component exists, then add it to its respective list.
            if root_metric is not None:
                #root_metrics.append(root_metric)
                if self.graph_capturable:
                    self.root_metrics.add_(root_metric)
                else:
                    self.root_metrics = self.root_metrics + root_metric
            if root_geometric_force is not None:
                #root_geometric_forces.append(root_geometric_force)
                if self.graph_capturable:
                    self.root_geometric_forces.add_(root_geometric_force)
                else:
                    self.root_geometric_forces = self.root_geometric_forces + root_geometric_force
            if root_potential_force is not None:
                #root_potential_forces.append(root_potential_force)
                if self.graph_capturable:
                    self.root_potential_forces.add_(root_potential_force)
                else:
                    self.root_potential_forces = self.root_potential_forces + root_potential_force
            if root_energy_metric is not None:
                #root_energy_metrics.append(root_energy_metric)
                if self.graph_capturable:
                    self.root_energy_metrics.add_(root_energy_metric)
                else:
                    self.root_energy_metrics = self.root_energy_metrics + root_energy_metric
            if root_energy_force is not None:
                #root_energy_forces.append(root_energy_force)
                if self.graph_capturable:
                    self.root_energy_forces.add_(root_energy_force)
                else:
                    self.root_energy_forces = self.root_energy_forces + root_energy_force
            if energy is not None:
                #energies.append(energy)
                if self.graph_capturable:
                    self.energies.add_(energy)
                else:
                    self.energies = self.energies + energy


        # Sum masses and forces across fabric terms.
        # NOTE: should always have a metric and geometric force
        #metric = torch.sum(torch.stack(root_metrics, 3), 3)
        #geometric_force = torch.sum(torch.stack(root_geometric_forces, 2), 2)
        metric = self.root_metrics
        geometric_force = self.root_geometric_forces

        # NOTE: potential force could be optional
#        if len(root_potential_forces) > 0:
#            potential_force = torch.sum(torch.stack(root_potential_forces, 2), 2)
#        else:
#            potential_force = None
        potential_force = self.root_potential_forces

        # Sum masses, forces, and energies across energy terms.
#        energy_metric = torch.sum(torch.stack(root_energy_metrics, 3), 3)
#        energy_force = torch.sum(torch.stack(root_energy_forces, 2), 2)
#        energy = torch.sum(torch.stack(energies, 1), 1).squeeze(1)

        energy_metric = self.root_energy_metrics
        energy_force = self.root_energy_forces
        energy = self.energies.squeeze(1)

        # Calculate resultant geometric acceleration, its energization, and
        # add potential force, and damping.
        
        # Calculate inverse of mass once and use in several places.
        # TODO: need to toggle choose between these two based on whether graph capture is set
        if self.graph_capturable:
            inverse_pd_matrix(metric, self.metric_inv, self.L, self.L_inv, self.device)
        else:
            self.metric_inv = torch.inverse(metric)
        #metric_inv = torch.zeros_like(metric)
        #L = torch.zeros_like(metric)
        #L_inv = torch.zeros_like(metric)
        #metric_inv = inverse_pd_matrix(metric, metric_inv, L, L_inv, self.device)

        # Calculate geometry acceleration.
        joint_accel = -torch.bmm(self.metric_inv,
                                 geometric_force.unsqueeze(2)).squeeze(2)

        # Calculate energization cofficient.
        scaling = (1./ (torch.bmm(torch.bmm(qd.unsqueeze(2).transpose(1,2), energy_metric),
                             qd.unsqueeze(2)) + 1e-6)).squeeze(2)
        alpha = -scaling * (torch.bmm(qd.unsqueeze(2).transpose(1,2),
                                      torch.bmm(energy_metric, joint_accel.unsqueeze(2)) +\
                                      energy_force.unsqueeze(2))).squeeze(2)

        # Writing everything directly in natural form.
        # Calculate mass times velocity once and it will be used in two places:
        #   1) energization
        #   2) additional cspace damping 
        mass_velocity = torch.bmm(metric, qd.unsqueeze(2)).squeeze(2)
        # Energized geometric force
        joint_force_energized =\
                geometric_force - alpha * mass_velocity
        force = joint_force_energized # set force to energized geometries

        # Add potential force if exists
        if potential_force is not None:
            force = force + potential_force
            
        # Add damping - this gain should always be specified in YAML file
        damping_gain = self.fabric_params['cspace_damping']['gain']
        force = force + damping_gain * mass_velocity
        
        # Add extra energization coefficient if specified. Applied along unit velocity
        if self.fabric_params.get('cspace_energization'):
            force = force + self.fabric_params['cspace_energization']['scalar'] * \
                    torch.bmm(metric, torch.nn.functional.normalize(qd).unsqueeze(2)).squeeze()
        
        if self.fabric_params['speed_control']['active']:
            speed_control_damping =\
                    (energy > self.fabric_params['speed_control']['energy_target']) *\
                    self.fabric_params['speed_control']['damping']
            force = force + speed_control_damping.unsqueeze(1) * mass_velocity

        # Acceleration limits.
        if self.fabric_params['joint_limits']['active'] is True:
            # First, update acceleration limits if needed
            #self.update_accel_limits(timestep)

            # Calculate acceleration
            qdd = -torch.bmm(self.metric_inv, force.unsqueeze(2)).squeeze(2)
#            metric_out.copy_(metric)
#            force_out.copy_(force)
#            metric_inv_out.copy_(metric_inv)
#
#            return (metric_out, force_out, metric_inv_out)

            # Scaled acceleration that respects acceleration limits and jerk limits.
            qdd_scaled = self.limit_accel_jerk(qdd)

            # Calculate associated force. 
            force = -torch.bmm(metric, qdd_scaled.unsqueeze(2)).squeeze(2)

#        metric_out.copy_(metric)
#        force_out.copy_(force)
#        metric_inv_out.copy_(metric_inv)

        #return (metric_out, force_out, metric_inv_out)
        return (metric, force, self.metric_inv)

    def eval_canonical(self, q, qd, timestep):
        """
        Calculates root acceleration.
        -----------------------------
        @param q: batched root position, bxn
        @param qd: batched root velocity, bxn
        @return qdd: batched root acceleration, bxn
        """

        # Get mass and force from natural eval and calculate acceleration from that.
        [M, f, M_inv] = self.eval_natural(q, qd)
        qdd = -torch.bmm(M_inv, f.unsqueeze(2))
        qdd = qdd.squeeze()

        return qdd

    def forward(self, q, qd, timestep): #, metric_out, force_out, metric_inv_out):
        """
        Evaluates fabric's combined mass and force.
        -----------------------------
        @param q: batched root position, bxn
        @param qd: batched root velocity, bxn
        @return metric: batched metric (mass) of the resulting fabric, bxnxn
        @return force: batched force of teh resulting fabric, bxn
        @return metric_inv: inverse of batched metric (mass), bxnxn
        """

        return self.eval_natural(q, qd, timestep) #, metric_out, force_out, metric_inv_out)

    def get_fabric_term(self, taskmap_name, fabric_name):
        """
        Returns the fabric term
        -----------------------------
        @param taskmap_name: str, name of the task map where the fabric term lives
        @param fabric_name: str, name of the fabric term
        @return fabric_term: fabric object
        """

        fabric_term = self.taskmap_containers[taskmap_name].get_fabric(fabric_name)

        return fabric_term
    
    def get_taskmap(self, taskmap_name):
        """
        Returns the task map.
        -----------------------------
        @param taskmap_name: str, name of the task map
        @return taskmap: taskmap object
        """
        taskmap = self.taskmap_containers[taskmap_name].taskmap

        return taskmap

    def get_taskmap_position(self, taskmap_name):
        """
        Returns the last evaluated taskmap position.
        -----------------------------
        @param taskmap_name: str, name of the task map
        @return x: batched taskmap position, bxm
        """

        return self.taskmap_containers[taskmap_name].x
    
    def get_taskmap_jacobian(self, taskmap_name):
        """
        Returns the last evaluated taskmap jacobian.
        -----------------------------
        @param taskmap_name: str, name of the task map
        @return x: batched taskmap jacobian, bxmxn
        """

        return self.taskmap_containers[taskmap_name].jac

    def get_fabric_term(self, taskmap_name, fabric_name):
        
        return self.taskmap_containers[taskmap_name].get_fabric(fabric_name)

    def allocate_scaled_accel(self):
        # If memory has not yet been allocated for the scaled acceleration
        # or if the batch size has changed, then re-allocate.
        #if self.allocated_data['qdd_scaled'] is None or\
        #   self.allocated_data['qdd_scaled'].shape[0] != qdd.shape[0]:
        self.allocated_data['qdd_scaled'] = wp.zeros(shape=(self.batch_size, self._num_joints), device=self.device)

    def limit_accel_jerk(self, qdd):

        # First check that number of joints set by qdd matches cspace dim of projection kernel
        assert qdd.shape[1] == self.allocated_data['cspace_dim'],\
            "Number of joints does not match number of values specified in yaml for acceleration and jerk limits."

        return AccelConstraint.apply(qdd, self.allocated_data)

    def get_joint_names(self):
        return self.joint_names

    @property
    def num_joints(self):
        return self._num_joints
    
    def load_robot(self, robot_dir_name=None, robot_name=None, batch_size=None):
        """
        Loads the robot model and kinematics.
        -----------------------------
        @param robot_name: name of robot (should be same as folder name and urdf name)
        @param batch_size: size of the batch
        """
        if robot_dir_name is not None:
            self.batch_size = batch_size

            # Load the robot.
            builder = wp.sim.ModelBuilder()

            robot_urdf_filename = get_robot_urdf_path(robot_dir_name, robot_name)
            # TODO: need to create an xform, which is a warp data type and pass it as a third argument here.
            initial_rotation = wp.quat(0., 0., 0., 1.)
            initial_position = wp.vec3(0., 0., 0.)
            initial_transform = wp.transform(initial_position, initial_rotation)

            # Make urdfpy robot so we can use it to access joint limits later
            self.urdfpy_robot = URDF.load(robot_urdf_filename)
            
            # Count number of active joints
            joints = self.urdfpy_robot.joints # this is a list
            self._num_joints = 0
            for i in range(len(joints)):
                # NOTE: We are only supporting revolute joints right now.
                if joints[i].joint_type == 'revolute':
                    self._num_joints += 1
                    self.joint_names.append(joints[i].name)

            # Convert to Warp object
            print('importing robot')
            wp.sim.parse_urdf(robot_urdf_filename, builder, initial_transform)
            
            print('finalizing model')
            self.model = builder.finalize(device=self.device)
            self.model.ground = True

        # Allocate memeory for scaled accelerations
        self.allocate_scaled_accel()

        # Pre-allocate some tensors
        self.root_metrics = torch.zeros(self.batch_size, self._num_joints, self._num_joints, device=self.device)
        self.root_geometric_forces = torch.zeros(self.batch_size, self._num_joints, device=self.device)
        self.root_potential_forces = torch.zeros(self.batch_size, self._num_joints, device=self.device)
        self.root_energy_metrics = torch.zeros(self.batch_size, self._num_joints, self._num_joints, device=self.device)
        self.root_energy_forces = torch.zeros(self.batch_size, self._num_joints, device=self.device)
        self.energies = torch.zeros(self.batch_size, 1, device=self.device)


        self.metric_inv = torch.zeros(self.batch_size, self._num_joints, self._num_joints, device=self.device)
        self.L = torch.zeros(self.batch_size, self._num_joints, self._num_joints, device=self.device)
        self.L_inv = torch.zeros(self.batch_size, self._num_joints, self._num_joints, device=self.device)

