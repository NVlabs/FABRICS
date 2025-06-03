# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os

import torch
import yaml
import warp as wp

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

@wp.kernel
def collision_response(
    # inputs
    robot_body_points: wp.array(dtype=wp.vec3, ndim=2),
    robot_body_point_vels: wp.array(dtype=wp.vec3, ndim=2),
    sphere_radius: wp.array(dtype=float, ndim=2),
    object_mesh: wp.array(dtype=wp.uint64, ndim=2),
    object_indicator: wp.array(dtype=wp.uint64, ndim=2),
    max_depth: float,
    min_depth: float,
    engage_depth: float,
    breakaway_depth: float,
    breakaway_velocity: float,
    metric_scalar: float,
    velocity_gate: bool,
    velocity_gate_sharpness: float,
    velocity_gate_offset: float,
    sphere_mesh_checked: wp.array(dtype=wp.int32, ndim=3),
    sphere_collision_matrix: wp.array(dtype=wp.int64, ndim=2),
    # outputs
    signed_sphere_distance: wp.array(dtype=float, ndim=2),
    base_acceleration: wp.array(dtype=wp.vec3, ndim=2),
    metric: wp.array(dtype=float, ndim=3)):

    batch_index, point_index, mesh_index, second_point_index = wp.tid()

    # Early out if env mesh does not exist 
    if object_indicator[batch_index, mesh_index] == 0:
        return

    # Now query closest distance between mesh object using this center face point and normal.
    d = float(0.0)              # hit distance along ray
    u = float(0.0)              # hit  face barycentric u
    v = float(0.0)              # hit  face barycentric u
    sign = float(0.0)           # hit face sign, value > 0 if ray hit front of face
    n = wp.vec3()               # hit face normal
    f = int(0)                  # hit face index
    inside = float(0.)          # < 0 if inside the mesh, 0 >= otherwise
    bary_u = float(0.)
    bary_v = float(0.)

    # If we haven't already queried the distance between the current body point and mesh, then do so
    # Checked if == 1
    # Not checked if == 0
    if sphere_mesh_checked[batch_index, point_index, mesh_index] == 0:
        # Immediately indicate that it's been checked
        wp.atomic_add(sphere_mesh_checked, batch_index, point_index, mesh_index, 1)

        sphere_center_point = robot_body_points[batch_index, point_index]
        
        # Query for distance between current body point and mesh
        got_dist = wp.mesh_query_point(object_mesh[batch_index, mesh_index],
                                       sphere_center_point, max_depth, inside, f, bary_u, bary_v)
        closest_point = wp.vec3()

        # If a positive return on distance check, then calculate response
        if got_dist:
            # First get the closest point on the mesh to the body point
            closest_point = wp.mesh_eval_position(object_mesh[batch_index, mesh_index],
                                                  f, bary_u, bary_v)
        
            # Direction from sphere to closest point on the mesh
            n = wp.normalize(closest_point - sphere_center_point)

            # Create signed distance. Positive is outside or 0, negative is inside.
            d = wp.length(closest_point - sphere_center_point)
            d = d * wp.sign(inside)

            # Project robot point velocity along direction to closest collision
            # Positive dir_vel means means moving away from the mesh
            dir_vel = -wp.dot(n, robot_body_point_vels[batch_index, point_index])

            # Safely save distance if it's the current smallest
            # Shave off sphere radius
            d_signed = d - sphere_radius[batch_index, point_index]
            # Clamp d so that it is postively bounded
            d = wp.clamp(d_signed, min_depth, max_depth)

            # Save off the signed distance
            wp.atomic_min(signed_sphere_distance, batch_index, point_index, d_signed)

            # Conditions for creating a metric and acceleration response:
            # 1) If the body point is too close to the mesh, then create response
            # 2) If the body point is moving towards the object while being sufficiently close,
            #    then, create response
            #if d_signed <= engage_depth or (dir_vel < breakaway_velocity and d_signed < breakaway_depth):
            #if True: # dir_vel <= breakaway_velocity or d_signed <= breakaway_depth:
            if d_signed <= engage_depth:

                # Add a weighted acceleration to the existing base acceleration response
                wp.atomic_add(base_acceleration, batch_index, point_index, metric_scalar * (1./d) * n)
                    
                # Create a rank deficient metric that cares about the direction towards collision
                # and scale according to a metric scalar parameter and barrier response
                point_metric = wp.mat33()
                if d_signed <= breakaway_depth:
                    point_metric = wp.outer(n, n) * metric_scalar * (1./d)
                else:
                    switch = float(1.)
                    if velocity_gate:
                        switch = 0.5 * (wp.tanh(-velocity_gate_sharpness*(dir_vel - velocity_gate_offset)) + 1.)
                    point_metric = wp.outer(n, n) * metric_scalar * (1./d) * switch
                
                # Add the metric to the existing metric response
                for i in range(3):
                    for j in range(3):
                        wp.atomic_add(metric, batch_index, point_index * 3 + i, point_index * 3 + j, point_metric[i,j])

    # Now check for body sphere collisions
    # Check the collision matrix. If value == 1, then calculate distance and collision response
    if sphere_collision_matrix[point_index, second_point_index] == 1:
        # Create a vector pointing from the current body point to another body point
        point1_to_point2 = robot_body_points[batch_index, second_point_index] -\
                           robot_body_points[batch_index, point_index]
        # Normalize this vector
        n = wp.normalize(point1_to_point2)

        # Calculate the distance between these two body spheres by calculating the distance between
        # the two body points and subtracting the radii of the spheres
        d_signed = wp.length(point1_to_point2) -\
                   sphere_radius[batch_index, point_index] -\
                   sphere_radius[batch_index, second_point_index]
            
        # Clamp d so that it is postively bounded
        d = wp.clamp(d_signed, min_depth, max_depth)
            
        # Project robot point velocity along direction to closest collision
        # Positive dir_vel means means moving away from the mesh
        dir_vel = -wp.dot(n, robot_body_point_vels[batch_index, point_index])

        # Save off the signed distance if it is the current smallest for this point
        wp.atomic_min(signed_sphere_distance, batch_index, point_index, d_signed)

        # Since the distance is between a pair of body spheres, update the signed distance
        # of the other point in the pair too
        wp.atomic_min(signed_sphere_distance, batch_index, second_point_index, d_signed)
            
        # Conditions for creating a metric and acceleration response:
        # 1) If the body point is too close to the other body point, then create response
        # 2) If the body point is moving towards the other body point while being sufficiently close
        #    then, create response
        #if d_signed <= engage_depth or (dir_vel > breakaway_velocity and d_signed < breakaway_depth):
        #if dir_vel <= breakaway_velocity or d_signed <= breakaway_depth:
        #if True:
        if d_signed <= engage_depth:
            # Add a weighted acceleration to the existing base acceleration response
            wp.atomic_add(base_acceleration, batch_index, point_index, metric_scalar * (1./d) * n)
            # Update second point of pair
            wp.atomic_add(base_acceleration, batch_index, second_point_index, metric_scalar * (1./d) * (-n))

            # Create a rank deficient metric that cares about the direction towards collision
            # and scale according to a metric scalar parameter and barrier response
            point_metric = wp.mat33()
            point_metric2 = wp.mat33()
            if d_signed <= breakaway_depth:
                point_metric = wp.outer(n, n) * metric_scalar * (1./d)
                point_metric2 = wp.outer(-n, -n) * metric_scalar * (1./d)
            else:
                switch = float(1.)
                if velocity_gate:
                    switch = 0.5 * (wp.tanh(-velocity_gate_sharpness*(dir_vel - velocity_gate_offset)) + 1.)
                point_metric = wp.outer(n, n) * metric_scalar * (1./d) * switch
                point_metric2 = wp.outer(-n, -n) * metric_scalar * (1./d) * switch
            
            # Add the metric to the existing metric response
            for i in range(3):
                for j in range(3):
                    wp.atomic_add(metric, batch_index, point_index * 3 + i, point_index * 3 + j, point_metric[i,j])
                    # Update second point of pair
                    wp.atomic_add(metric, batch_index, second_point_index * 3 + i, second_point_index * 3 + j, point_metric2[i,j])

def eval_collision_response_func(point_positions, point_velocities, allocated_data):
    """
    This function does some data preparation like zeroing out certain signals and calls 
    the collision response kernel
    ----------------------------------------------
    :param point_positions: b x num_body_points x 3 warp float array containing positions of
                            body points
    :param point_velocities: b x num_body_points x 3 warp float array containing velocities
                             of body points
    :param allocated_data: dict of various data structures for the collision response kernel
    """

    # Size of batch
    batch_size = allocated_data['batch_size']
    # Number of points on robot body
    num_points = allocated_data['num_points']
    # Number of object meshes in world
    num_meshes = allocated_data['object_mesh'].shape[1]

    # Clear the previous values for distance
    #signed_sphere_distance = 0. * wp.torch.to_torch(allocated_data['signed_sphere_distance']) + 1e6
    signed_sphere_distance = wp.torch.to_torch(allocated_data['signed_sphere_distance'])
    signed_sphere_distance.zero_().add_(1e6)
    allocated_data['signed_sphere_distance'] = wp.torch.from_torch(signed_sphere_distance)

    # Zero out metric and base acceleration
    #metric = 0. * wp.torch.to_torch(allocated_data['metric'])
    metric = wp.torch.to_torch(allocated_data['metric'])
    metric.zero_()
    allocated_data['metric'] = wp.torch.from_torch(metric)

    #base_acceleration = 0. * wp.torch.to_torch(allocated_data['base_acceleration'])
    base_acceleration = wp.torch.to_torch(allocated_data['base_acceleration'])
    base_acceleration.zero_()
    allocated_data['base_acceleration'] = wp.torch.from_torch(base_acceleration, dtype=wp.vec3)

    # Allocate or re-allocate sphere_mesh_checked data if needed
    if allocated_data['sphere_mesh_checked'] is None or\
       allocated_data['sphere_mesh_checked'].shape[2] != num_meshes:
           allocated_data['sphere_mesh_checked'] =\
               wp.zeros(shape=(batch_size, num_points, num_meshes),
                        dtype=wp.int32,
                        device=allocated_data['device'])
    
    # Reset whether meshes were checked against robot for collision
    #sphere_mesh_checked = 0 * wp.torch.to_torch(allocated_data['sphere_mesh_checked'])
    sphere_mesh_checked = wp.torch.to_torch(allocated_data['sphere_mesh_checked'])
    sphere_mesh_checked.zero_()
    allocated_data['sphere_mesh_checked'] = wp.torch.from_torch(sphere_mesh_checked)

    # Invoke the collision repsonse kernel
    wp.launch(kernel=collision_response,
              dim=[batch_size, num_points, num_meshes, num_points],
              inputs=[
                  point_positions,
                  point_velocities,
                  allocated_data['sphere_radius'],
                  allocated_data['object_mesh'],
                  allocated_data['object_indicator'],
                  allocated_data['max_depth'],
                  allocated_data['min_depth'],
                  allocated_data['engage_depth'],
                  allocated_data['breakaway_depth'],
                  allocated_data['breakaway_velocity'],
                  allocated_data['metric_scalar'],
                  allocated_data['velocity_gate'],
                  allocated_data['velocity_gate_sharpness'],
                  allocated_data['velocity_gate_offset'],
                  allocated_data['sphere_mesh_checked'],
                  allocated_data['sphere_collision_matrix']
                  ],
              outputs=[
                  allocated_data['signed_sphere_distance'],
                  allocated_data['base_acceleration'],
                  allocated_data['metric']
                  ],
              device=allocated_data['device'])

# Define PyTorch autograd op to invoke collision response kernel, performing the
# forward and backward pass
class CollisionResponse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, point_positions, point_velocities, allocated_data):
        """
        :param point_positions: b x (3 * num_body_points) Pytorch tensor of 3D body point positions
        :param point_velocities: b x (3 * num_body_points) Pytorch tensor of 3D body point
               velocities
        :param allocated_data: dict of various data structures for the collision response kernel
        """

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.point_positions = wp.torch.from_torch(point_positions, dtype=wp.vec3)
        ctx.point_velocities = wp.torch.from_torch(point_velocities, dtype=wp.vec3)
        ctx.allocated_data = allocated_data
        
        with ctx.tape:
            eval_collision_response_func(
                ctx.point_positions,
                ctx.point_velocities,
                ctx.allocated_data)
        
        return (wp.torch.to_torch(ctx.allocated_data['signed_sphere_distance']),
                wp.torch.to_torch(ctx.allocated_data['base_acceleration']),
                wp.torch.to_torch(ctx.allocated_data['metric']))

    @staticmethod
    def backward(ctx, adj_signed_sphere_distance, adj_base_acceleration, adj_metric):
        """
        :param adj_signed_sphere_distance: Pytorch tensor gradient of signed sphere distance
        :param adj_base_acceleration: Pytorch tensor gradient of base acceleration
        :param adj_metric: Pytorch tensor gradient of metric
        :return: Pytorch tensor gradients with respect to body point position and velocity
        """
        # Map incoming Torch grads to the output variable grads
        ctx.allocated_data['signed_sphere_distance'].grad = wp.torch.from_torch(adj_signed_sphere_distance)
        ctx.allocated_data['base_acceleration'].grad = wp.torch.from_torch(adj_base_acceleration)
        ctx.allocated_data['metric'].grad = wp.torch.from_torch(adj_metric)
    
        # Calculate gradients
        ctx.tape.backward()

        wp.synchronize_device()

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.point_positions]),
                wp.torch.to_torch(ctx.tape.gradients[ctx.point_velocities]),
                None)

class BaseFabricRepulsion():
    """
    Calculates base metric, acceleration, and signed distance for every body sphere
    origin on the robot.
    """
    def __init__(self, params, batch_size, sphere_radius, collision_matrix, device):
        """
        Constructor.
        ------------------------------------------
        :param params: dict of collision parameters and body point frame data
        :param batch_size: int, size of batch
        :param sphere_radius: bxn Pytorch tensor, b is batch size, n is number of body spheres 
        :param collision_matrix: nxn Pytorch tensor, n is number of body spheres
        """
        # Some signals to calculate and maintain
        self._num_points = sphere_radius.shape[1]
        self._base_metric = None
        self._accel_dir = None
        self._signed_distance = None
        self._in_collision = None
        
        # Dictionary of various signals required for repulsion calculations.
        self.allocated_data =\
            { 'batch_size': batch_size,
              'num_points': self._num_points,
              'robot_body_points': wp.zeros(shape=(batch_size, self._num_points), dtype=wp.vec3, device=device),
              'robot_body_point_vels': wp.zeros(shape=(batch_size, self._num_points), dtype=wp.vec3, device=device),
              'sphere_radius': wp.torch.from_torch(sphere_radius),
              'object_mesh': None,
              'object_indicator': None,
              'max_depth': params['max_depth'],
              'min_depth': params['min_depth'],
              'engage_depth': params['engage_depth'],
              'breakaway_depth': params['breakaway_depth'],
              'breakaway_velocity': params['breakaway_velocity'],
              'metric_scalar': params['metric_scalar'],
              'velocity_gate': params['velocity_gate'],
              'velocity_gate_sharpness': params['velocity_gate_sharpness'],
              'velocity_gate_offset': params['velocity_gate_offset'],
              'sphere_mesh_checked': None,
              'sphere_collision_matrix': wp.torch.from_torch(collision_matrix),
              'signed_sphere_distance': wp.zeros(shape=(batch_size, self._num_points), dtype=float, device=device),
              'base_acceleration': wp.zeros(shape=(batch_size, self._num_points), dtype=wp.vec3, device=device),
              'metric': wp.zeros(shape=(batch_size, self._num_points * 3, self._num_points * 3), dtype=float, device=device),
              'device': device }

    def calculate_response(self, x, xd, object_mesh_ids, object_indicator):
        """
        Invokes the collision reponse kernel and calculates base metric response, acceleration
        direction, and collision status.
        ------------------------------------------
        :param x: bx(3*n) Pytorch tensor that the 3D location of every body sphere origin stacked
        :param xd: bx(3*n) Pytorch tensor that the 3D velocity of every body sphere origin stacked
        :param object_mesh_ids: 2D int Warp array referencing object meshes
        :param object_indicator: 2D Warp array of type uint64, indicating the presence
                                 of a Warp mesh in object_ids at corresponding index
                                 0=no mesh, 1=mesh
        """
        batch_size = self.allocated_data['batch_size']
        num_points = self.allocated_data['num_points']
        self.allocated_data['object_mesh'] = object_mesh_ids
        self.allocated_data['object_indicator'] = object_indicator
        # Launch the collision response kernel
        (signed_distance, base_acceleration, metric) =\
            CollisionResponse.apply(x.reshape(batch_size, num_points, 3),
                                    xd.reshape(batch_size, num_points, 3),
                                    self.allocated_data)
        if self._signed_distance is None:
            self._signed_distance = signed_distance
        else:
            self._signed_distance.copy_(signed_distance)

        # Set the base metric response equal to the returned metric
        metric_norms = torch.linalg.norm(metric, dim=(1,2)).unsqueeze(1).unsqueeze(2)

        # NOrmalize the metric in a numerically robust way
        eps = 1e-6
        if self._base_metric is None:
            self._base_metric = (metric / (metric_norms + eps))
        else:
            self._base_metric.copy_(metric / (metric_norms + eps))
        
        # Normalize the acceleration to get a direction of unit magnitude
        if self._accel_dir is None:
            self._accel_dir = torch.nn.functional.normalize(base_acceleration, dim=-1)
        else:
            self._accel_dir.copy_(torch.nn.functional.normalize(base_acceleration, dim=-1))
    
        # Report collision detection. True if any negative distance is found across the entire
        # robot body
        _in_collision, _ = (self._signed_distance < 0.).max(dim=-1)
        if self._in_collision is None:
            self._in_collision = _in_collision
        else:
            self._in_collision.copy_(_in_collision)

    @property
    def collision_status(self):
        return self._in_collision

    @property
    def base_metric(self):
        return self._base_metric

    @property
    def accel_dir(self):
        return self._accel_dir

    @property
    def signed_distance(self):
        return self._signed_distance

    @property
    def num_points(self):
        return self._num_points
            
class BodySphereRepulsion(BaseFabricTerm):
    """
    Implements fabric repulsion term between body spheres and environment.
    """
    def __init__(self, is_forcing_policy, params, batch_size, sphere_radius,
                 collision_matrix, device, graph_capturable):
        """
        Constructor.
        -----------------------------
        :param is_forcing_policy: bool, indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        :param params: dict of collision parameters and body point frame data
        :param batch_size: int, size of batch
        :param sphere_radius: bxn Pytorch tensor, b is batch size, n is number of body spheres 
        :param collision_matrix: nxn Pytorch tensor, n is number of body spheres
        :param device: str, device id, e.g., 'cuda:0'
        """
        self.batch_size = batch_size
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)

        num_points = sphere_radius.shape[1]
        self.indicesx = torch.tensor([i * 3 for i in range(0, num_points)], device=self.device)
        self.indicesy = torch.tensor([i * 3 + 1 for i in range(0, num_points)], device=self.device)
        self.indicesz = torch.tensor([i * 3 + 2 for i in range(0, num_points)], device=self.device)


        self.expanded_dist = None

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        :param x: bx(3*n) Pytorch tensor that the 3D location of every body sphere origin stacked
        :param xd: bx(3*n) Pytorch tensor that the 3D velocity of every body sphere origin stacked
        :param features: instantiated BaseFabricRepulsion object
        :return metric: bx(3*n)x(3*n) Pytorch tensor that is the fabric metric
        """
        if self.metric is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.expanded_dist = torch.zeros(self.batch_size, 3 * features.num_points, device=self.device)

        if self.expanded_dist.ndim == 3:
            self.expanded_dist = self.expanded_dist.squeeze(2)
        
        if self.graph_capturable:
            self.metric.zero_().detach_()
            self.force.zero_().detach_()
            self.expanded_dist.zero_().detach()
        else:
            self.metric = torch.zeros_like(self.metric)
            self.force = torch.zeros_like(self.force)
            self.expanded_dist = torch.zeros_like(self.expanded_dist)

        if features is not None:
            unsigned_dist = torch.clamp(features.signed_distance, min=self.params['rescaled_min_dist'])
            #self.expanded_dist = torch.zeros(self.batch_size, 3 * features.num_points, device=self.device)
            self.expanded_dist[:, self.indicesx] = unsigned_dist
            self.expanded_dist[:, self.indicesy] = unsigned_dist
            self.expanded_dist[:, self.indicesz] = unsigned_dist
            self.expanded_dist = self.expanded_dist.unsqueeze(2)

            if self.is_forcing_policy:
                # Provide additional scaling for forcing fabric term on metric
                if self.graph_capturable:
                    self.metric.copy_((self.params['forcing_metric_scalar'] / self.expanded_dist**2) *\
                                  features.base_metric)
                else:
                    self.metric = (self.params['forcing_metric_scalar'] / self.expanded_dist**2) *\
                                   features.base_metric
            else:
                # Provide additional scaling for geometric fabric term on metric
                if self.graph_capturable:
                    self.metric.copy_((self.params['geom_metric_scalar'] / self.expanded_dist**2) *\
                                  features.base_metric)
                else:
                    self.metric = (self.params['geom_metric_scalar'] / self.expanded_dist**2) *\
                                  features.base_metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this attractor term.
        -----------------------------
        :param x: bx(3*n) Pytorch tensor that the 3D location of every body sphere origin stacked
        :param xd: bx(3*n) Pytorch tensor that the 3D velocity of every body sphere origin stacked
        :param features: instantiated BaseFabricRepulsion object
        :return force: bx(3*n) Pytorch tensor that is the fabric force
        """
        # Check to see if no target is set. If true, then
        # set acceleration to zeros.
        if features is None:
            xdd = torch.zeros(x.shape, device=self.device)
        else: 
            # Extract acceleration direction and reshape into acceleration of shape
            # in this concatenated space.
            accel_dir = features.accel_dir.reshape(self.batch_size,
                                                   features.num_points * 3)

            if self.is_forcing_policy:
                # Scale the acceleration by some positive value and add damping
                xdd = -self.params['constant_accel'] * accel_dir -\
                      self.params['damping_gain'] * xd
            else:
                # Scale the acceleration by some positive value and the inner product of
                # velocity to create a geometry
                xdd_not_hd2 = -self.params['constant_accel_geom'] * accel_dir
                vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
                xdd = vel_squared * xdd_not_hd2
        
        # Convert to force.
        if self.graph_capturable:
            self.force.copy_(-torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2))
        else:
            self.force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

