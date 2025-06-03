# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a map to a 3D point on the robot body.
"""

import os
import torch
import numpy as np

import warp as wp
import warp.torch

from fabrics_sim.taskmaps.maps_base import BaseMap

"""
Warp kernel for calculating pose of a particular frame on the robot expressed as an origin
and three rotational axis of unit length.
"""
@wp.kernel
def eval_pose(
    articulation_start: wp.array(dtype=int),
    articulation_mask: wp.array(dtype=int), # used to enable / disable FK for an articulation, if None then treat all as enabled
    joint_q: wp.array(dtype=float, ndim=2),
    joint_q_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform, ndim=2), # Trying to make this an input so we don't have to calculate derivates for it. Not sure if it'll work.
    target_body_index: int, # Which body you want to get the pose for
    num_bodies: int,
    # outputs
    body_pos_o: wp.array(dtype=wp.vec3),
    body_axis_x: wp.array(dtype=wp.vec3),
    body_axis_y: wp.array(dtype=wp.vec3),
    body_axis_z: wp.array(dtype=wp.vec3),
    jacobian_pos_o: wp.array(dtype=float),
    jacobian_axis_x: wp.array(dtype=float),
    jacobian_axis_y: wp.array(dtype=float),
    jacobian_axis_z: wp.array(dtype=float)):

    batch_index = wp.tid()

#    # early out if disabling FK for this articulation
#    if (articulation_mask):
#        if (articulation_mask[tid]==0):
#            return

    joint_start = articulation_start[0]
    joint_end = articulation_start[1]

    for i in range(joint_start, joint_end):

        parent = joint_parent[i]
        X_wp = wp.transform_identity()

        if (parent >= 0):
            X_wp = body_q[batch_index, parent]

        # compute transform across the joint
        type = joint_type[i]
        axis = joint_axis[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]  
        
        q_start = joint_q_start[i]

        if type == wp.sim.JOINT_REVOLUTE:

            q = joint_q[batch_index, q_start]

            X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))

        if type == wp.sim.JOINT_FIXED:
            
            X_jc = wp.transform_identity()

        X_wj = X_wp*X_pj
        X_wc = X_wj*X_jc

        body_q[batch_index, i] = X_wc

    # Cycle through a pull out four 3D points on the target body coordinate system.
    body_transform = body_q[batch_index, target_body_index]
    body_pos_o[batch_index] = wp.transform_get_translation(body_transform)
    # Extract rotational axes
    q0 = body_transform[6]
    q1 = body_transform[3]
    q2 = body_transform[4]
    q3 = body_transform[5]
    # Convert quaternion to rotational axes and insert.
    body_axis_x[batch_index] = wp.vec3(2. * (q0 * q0 + q1 * q1) - 1., 2. * (q1 * q2 + q0 * q3), 2. * (q1 * q3 - q0 * q2))
    body_axis_y[batch_index] = wp.vec3(2. * (q1 * q2 - q0 * q3), 2. * (q0 * q0 + q2 * q2) - 1., 2. * (q2 * q3 + q0 * q1))
    body_axis_z[batch_index] = wp.vec3(2. * (q1 * q3 + q0 * q2), 2. * (q2 * q3 - q0 * q1), 2. * (q0 * q0 + q3 * q3) - 1.)

    # Create unit offset in x-coordinate
    x_offset = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
    body_pos_x = wp.transform_get_translation(body_transform * x_offset)
    
    # Create unit offset in y-coordinate
    y_offset = wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity())
    body_pos_y = wp.transform_get_translation(body_transform * y_offset)
    
    # Create unit offset in z-coordinate
    z_offset = wp.transform(wp.vec3(0.0, 0.0, 1.), wp.quat_identity())
    body_pos_z = wp.transform_get_translation(body_transform * z_offset)

    # Build Jacobian
    # Calculate the number of actively controlled joints
    num_joints = int(0)
    for i in range(joint_start, joint_end):
        if joint_type[i] == wp.sim.JOINT_REVOLUTE:
            num_joints += 1
    joint_index = int(0)
    for i in range(joint_start, joint_end):
        # Only build jacobian components for an actual active joint. In this case, we just check for revolute.
        if joint_type[i] == wp.sim.JOINT_REVOLUTE:

            # Extract z rotation axis for current joint
            q0 = body_q[batch_index, i][6]
            q1 = body_q[batch_index, i][3]
            q2 = body_q[batch_index, i][4]
            q3 = body_q[batch_index, i][5]

            z_axis = wp.vec3(2. * (q1 * q3 + q0 * q2), 2. * (q2 * q3 - q0 * q1), 2. * (q0 * q0 + q3 * q3) - 1.)

            # Calculate Jacobian for the origin of target body---------------------------------------------
            # Calcualte joint position diff
            joint_pos_diff = body_pos_o[batch_index] - wp.transform_get_translation(body_q[batch_index, i])

            # Cross product of z_axis joint pos diff
            x = z_axis
            y = joint_pos_diff
            
            cross_result_x = x[1]*y[2] - x[2]*y[1]
            cross_result_y = -(x[0]*y[2] - x[2]*y[0])
            cross_result_z = x[0]*y[1] - x[1]*y[0]

            # Insert values for the Jacobian of the origin
            jacobian_pos_o[batch_index*num_joints*3 + 0*num_joints + joint_index] = cross_result_x
            jacobian_pos_o[batch_index*num_joints*3 + 1*num_joints + joint_index] = cross_result_y
            jacobian_pos_o[batch_index*num_joints*3 + 2*num_joints + joint_index] = cross_result_z
            
            # Calculate Jacobian for the x axis of target body---------------------------------------------
            # Calcualte joint position diff
            joint_pos_diff = body_pos_x - body_pos_o[batch_index]

            # Cross product of z_axis joint pos diff
            y = joint_pos_diff
            
            cross_result_x = x[1]*y[2] - x[2]*y[1]
            cross_result_y = -(x[0]*y[2] - x[2]*y[0])
            cross_result_z = x[0]*y[1] - x[1]*y[0]

            # Insert values for the Jacobian of the x axis
            jacobian_axis_x[batch_index*num_joints*3 + 0*num_joints + joint_index] = cross_result_x
            jacobian_axis_x[batch_index*num_joints*3 + 1*num_joints + joint_index] = cross_result_y
            jacobian_axis_x[batch_index*num_joints*3 + 2*num_joints + joint_index] = cross_result_z
            
            # Calculate Jacobian for the y axis of target body---------------------------------------------
            joint_pos_diff = body_pos_y - body_pos_o[batch_index]

            # Cross product of z_axis joint pos diff
            y = joint_pos_diff
            
            cross_result_x = x[1]*y[2] - x[2]*y[1]
            cross_result_y = -(x[0]*y[2] - x[2]*y[0])
            cross_result_z = x[0]*y[1] - x[1]*y[0]

            # Insert values for the Jacobian of the x axis
            jacobian_axis_y[batch_index*num_joints*3 + 0*num_joints + joint_index] = cross_result_x
            jacobian_axis_y[batch_index*num_joints*3 + 1*num_joints + joint_index] = cross_result_y
            jacobian_axis_y[batch_index*num_joints*3 + 2*num_joints + joint_index] = cross_result_z
            
            # Calculate Jacobian for the z axis of target body---------------------------------------------
            joint_pos_diff = body_pos_z - body_pos_o[batch_index]

            # Cross product of z_axis joint pos diff
            y = joint_pos_diff
            
            cross_result_x = x[1]*y[2] - x[2]*y[1]
            cross_result_y = -(x[0]*y[2] - x[2]*y[0])
            cross_result_z = x[0]*y[1] - x[1]*y[0]

            # Insert values for the Jacobian of the x axis
            jacobian_axis_z[batch_index*num_joints*3 + 0*num_joints + joint_index] = cross_result_x
            jacobian_axis_z[batch_index*num_joints*3 + 1*num_joints + joint_index] = cross_result_y
            jacobian_axis_z[batch_index*num_joints*3 + 2*num_joints + joint_index] = cross_result_z

            joint_index += 1

def eval_pose_func(model, joint_q, mask, fk_kernel_data):

    wp.launch(kernel=eval_pose,
              dim=fk_kernel_data['batch_size'],
              inputs=[    
                  model.articulation_start,
                  mask,
                  joint_q,
                  model.joint_q_start,
                  model.joint_type,
                  model.joint_parent,
                  model.joint_X_p,
                  model.joint_X_c,
                  fk_kernel_data['joint_axes'],
                  fk_kernel_data['body_q'],
                  fk_kernel_data['target_body_index'],
                  fk_kernel_data['num_bodies']],
              outputs=[
                  fk_kernel_data['body_pos_o'],
                  fk_kernel_data['body_axis_x'],
                  fk_kernel_data['body_axis_y'],
                  fk_kernel_data['body_axis_z'],
                  fk_kernel_data['jacobian_pos_o'],
                  fk_kernel_data['jacobian_axis_x'],
                  fk_kernel_data['jacobian_axis_y'],
                  fk_kernel_data['jacobian_axis_z']],
              device=model.device)

# Define PyTorch autograd op to wrap foward kinematics
# function.
class ForwardKinematics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, joint_q, model, allocated_state):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.model = model
        ctx.joint_q = wp.torch.from_torch(joint_q)
        ctx.allocated_state = allocated_state
        
        with ctx.tape:
            eval_pose_func(
                ctx.model,
                ctx.joint_q,
                None,
                ctx.allocated_state)
        
        return (wp.torch.to_torch(ctx.allocated_state['body_pos_o']),
                wp.torch.to_torch(ctx.allocated_state['body_axis_x']),
                wp.torch.to_torch(ctx.allocated_state['body_axis_y']),
                wp.torch.to_torch(ctx.allocated_state['body_axis_z']),
                wp.torch.to_torch(ctx.allocated_state['jacobian_pos_o']),
                wp.torch.to_torch(ctx.allocated_state['jacobian_axis_x']),
                wp.torch.to_torch(ctx.allocated_state['jacobian_axis_y']),
                wp.torch.to_torch(ctx.allocated_state['jacobian_axis_z']))

    @staticmethod
    def backward(ctx, adj_body_pos_o, adj_body_axis_x, adj_body_axis_y, adj_body_axis_z,
                 adj_jacobian_pos_o, adj_jacobian_axis_x, adj_jacobian_axis_y,
                 adj_jacobian_axis_z):

        # Map incoming Torch grads to our output variables
        grads = { ctx.allocated_state['body_pos_o']:
                      wp.torch.from_torch(adj_body_pos_o, dtype=wp.vec3),
                  ctx.allocated_state['body_axis_x']:
                      wp.torch.from_torch(adj_body_axis_x, dtype=wp.vec3),
                  ctx.allocated_state['body_axis_y']:
                      wp.torch.from_torch(adj_body_axis_y, dtype=wp.vec3),
                  ctx.allocated_state['body_axis_z']:
                      wp.torch.from_torch(adj_body_axis_z, dtype=wp.vec3),
                  ctx.allocated_state['jacobian_pos_o']:
                      wp.torch.from_torch(adj_jacobian_pos_o, dtype=float),
                  ctx.allocated_state['jacobian_axis_x']:
                      wp.torch.from_torch(adj_jacobian_axis_x, dtype=float),
                  ctx.allocated_state['jacobian_axis_y']:
                      wp.torch.from_torch(adj_jacobian_axis_y, dtype=float),
                  ctx.allocated_state['jacobian_axis_z']:
                      wp.torch.from_torch(adj_jacobian_axis_z, dtype=float) }

        # Calculate gradients
        ctx.tape.backward(grads=grads)

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.joint_q]),
                None,
                None)

class PoseMap(BaseMap):
    def __init__(self, model, batch_size, target_body_index, device):
        """
        Initializes the map to a coordinate system on the robot.
        -----------------------------
        @param model: warp model of the robot
        @param batch_size: int, batch size of tensors for parallel evals
        @param target_body_index: int, index of coordinate system on robot for map
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        self.model = model
        self.target_body_index = target_body_index
        
        # Find number of revolute joints
        # TODO: this and the kernel currently only supports revolute joints. add other joint types
        self.num_active_joints = 0
        for joint_type in list(self.model.joint_type.numpy()):
            if joint_type == 1:
                self.num_active_joints += 1

        self.num_bodies = self.model.body_count
        self.allocate_variables(batch_size)

    def allocate_variables(self, batch_size):
        self.batch_size = batch_size
    
        # NOTE: Warp > 0.7.2 made a change so that fixed joints are removed
        # from self.model.joint_axis. This breaks our kinematics kernels. The fix
        # is to reinject [0,0,0] (fixed joint axis) for every fixed joint at the right
        # location in the joint_axis array.

        # Get the total number of joints including fixed joints
        num_joints = self.model.joint_type.shape[0]

        # Allocate a 2D numpy array with 0s
        local_joint_axes_np = np.zeros((num_joints, 3))

        # Create numpy array for non-fixed joint axis data
        local_articulated_joint_axes_np = self.model.joint_axis.numpy()

        # Find where joint_types are not equal to 3 (3 indicates a fixed joint) 
        joint_types_np = self.model.joint_type.numpy()
        non_fixed_joint_indices = np.where(joint_types_np!=3)[0]

        # Inject non-fixed joint axis data into the full joint axes data array
        local_joint_axes_np[non_fixed_joint_indices] = local_articulated_joint_axes_np

        # Convert the numpy joint axis data to warp array
        local_joint_axes = wp.array(local_joint_axes_np, dtype=wp.vec3, device=self.device)

        # Allocate various input and output arrays/data.
        self.allocated_state =\
            { 
              'batch_size': batch_size,
              'body_q': wp.zeros((self.batch_size, self.num_bodies), dtype=wp.transform, device=self.device),
              'joint_axes': local_joint_axes,
              'target_body_index': self.target_body_index,
              'num_bodies': self.num_bodies,
              'body_pos_o': wp.zeros(batch_size, dtype=wp.vec3, device=self.model.device),
              'body_axis_x': wp.zeros(batch_size, dtype=wp.vec3, device=self.model.device),
              'body_axis_y': wp.zeros(batch_size, dtype=wp.vec3, device=self.model.device),
              'body_axis_z': wp.zeros(batch_size, dtype=wp.vec3, device=self.model.device),
              'jacobian_pos_o': wp.zeros(batch_size * 3 * self.num_active_joints, dtype=float, device=self.device),
              'jacobian_axis_x': wp.zeros(batch_size * 3 * self.num_active_joints, dtype=float, device=self.device),
              'jacobian_axis_y': wp.zeros(batch_size * 3 * self.num_active_joints, dtype=float, device=self.device),
              'jacobian_axis_z': wp.zeros(batch_size * 3 * self.num_active_joints, dtype=float, device=self.device)
            }

    def forward_position(self, q, features):

        # Check if batch size has changed and resize memory allocations if necessary.
        if q.shape[0] != self.batch_size:
            self.allocate_variables(q.shape[0])

        # Forward kinematics for all bodies along robot.
        (body_pos_o, body_axis_x, body_axis_y, body_axis_z,\
         jacobian_pos_o, jacobian_axis_x, jacobian_axis_y, jacobian_axis_z) =\
            ForwardKinematics.apply(q, self.model, self.allocated_state)

        # Create poses of size batch x 12
        pose = torch.cat((body_pos_o, body_axis_x, body_axis_y, body_axis_z), 1)

        # Create jacobians of size batch x 12 x num_active_joints
        jacobian_pos_o = jacobian_pos_o.reshape((self.batch_size, 3, self.num_active_joints))
        jacobian_axis_x = jacobian_axis_x.reshape((self.batch_size, 3, self.num_active_joints))
        jacobian_axis_y = jacobian_axis_y.reshape((self.batch_size, 3, self.num_active_joints))
        jacobian_axis_z = jacobian_axis_z.reshape((self.batch_size, 3, self.num_active_joints))
        jacobian = torch.cat((jacobian_pos_o, jacobian_axis_x, jacobian_axis_y, jacobian_axis_z), 1)
        
        return (pose, jacobian)




