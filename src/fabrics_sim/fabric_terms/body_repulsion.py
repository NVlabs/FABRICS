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
import time
import numpy as np

import warp as wp
import ghalton

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

@wp.func
def rotate_normal(ray_dir: wp.vec3,
                  angle_x: float,
                  angle_z: float):
    """
    This rotates the ray direction vector by a certain degree about the x axis
    and a certain degree about the z axis. This allows you to change the
    ray direction so we can ray cast within a cone set by the max angle.
    """

    x_rot = wp.mat33(1., 0., 0.,
                     0., cos(angle_x), -sin(angle_x),
                     0., sin(angle_x), cos(angle_x))

    z_rot = wp.mat33(cos(angle_z), -sin(angle_z), 0.,
                     sin(angle_z), cos(angle_z), 0.,
                     0., 0., 1.)

    return wp.mul(wp.mul(x_rot, z_rot), ray_dir)

@wp.kernel
def eval_repulsion(
    # inputs
    robot_body_pose: wp.array(dtype=float, ndim=2),
    robot_body_velocity: wp.array(dtype=float, ndim=2),
    robot_body_indices: wp.array(dtype=int),
    robot_body_points: wp.array(dtype=wp.vec3),
    object_mesh: wp.uint64,
    max_depth: float,
    num_points: int,
    num_faces: int,
    ray_angles: wp.array(dtype=float, ndim=2),
    num_rays: int,
    # outputs
    collision_state: wp.array(dtype=int, ndim=3),
    metrics: wp.array(dtype=wp.mat33, ndim=3),
    forces: wp.array(dtype=wp.vec3, ndim=3),
    linear_maps: wp.array(dtype=float, ndim=4),
    ray_hit_points: wp.array(dtype=wp.vec3, ndim=2)):

    # tid is element of [0, batch size * number of faces)
    tid, ray_index = wp.tid()
    
    batch_number = wp.div(tid, num_faces)
    face_index = wp.mod(tid, num_faces)
    
    # Get the three points of a triangular face
    point1 = robot_body_points[robot_body_indices[face_index * 3]]
    point2 = robot_body_points[robot_body_indices[face_index * 3 + 1]]
    point3 = robot_body_points[robot_body_indices[face_index * 3 + 2]]
    
    # Calculate the centroid expressed in body-centric coordinates
    center_point_body =\
        wp.div(wp.add(wp.add(point1, point2), point3), 3.)

    # Calculate the 3D velocity of this center body point
    center_point_vel = wp.vec3(robot_body_velocity[batch_number, 0 + 3] * center_point_body[0] +\
                               robot_body_velocity[batch_number, 0 + 6] * center_point_body[1] +\
                               robot_body_velocity[batch_number, 0 + 9] * center_point_body[2] +\
                               robot_body_velocity[batch_number, 0],
                               robot_body_velocity[batch_number, 1 + 3] * center_point_body[0] +\
                               robot_body_velocity[batch_number, 1 + 6] * center_point_body[1] +\
                               robot_body_velocity[batch_number, 1 + 9] * center_point_body[2] +\
                               robot_body_velocity[batch_number, 1],
                               robot_body_velocity[batch_number, 2 + 3] * center_point_body[0] +\
                               robot_body_velocity[batch_number, 2 + 6] * center_point_body[1] +\
                               robot_body_velocity[batch_number, 2 + 9] * center_point_body[2] +\
                               robot_body_velocity[batch_number, 2])

    # Transform the points using the current pose of the body
    point1_transformed = wp.vec3(robot_body_pose[batch_number, 0 + 3] * point1[0] +\
                                 robot_body_pose[batch_number, 0 + 6] * point1[1] +\
                                 robot_body_pose[batch_number, 0 + 9] * point1[2] +\
                                 robot_body_pose[batch_number, 0],
                                 robot_body_pose[batch_number, 1 + 3] * point1[0] +\
                                 robot_body_pose[batch_number, 1 + 6] * point1[1] +\
                                 robot_body_pose[batch_number, 1 + 9] * point1[2] +\
                                 robot_body_pose[batch_number, 1],
                                 robot_body_pose[batch_number, 2 + 3] * point1[0] +\
                                 robot_body_pose[batch_number, 2 + 6] * point1[1] +\
                                 robot_body_pose[batch_number, 2 + 9] * point1[2] +\
                                 robot_body_pose[batch_number, 2])

    point2_transformed = wp.vec3(robot_body_pose[batch_number, 0 + 3] * point2[0] +\
                                 robot_body_pose[batch_number, 0 + 6] * point2[1] +\
                                 robot_body_pose[batch_number, 0 + 9] * point2[2] +\
                                 robot_body_pose[batch_number, 0],
                                 robot_body_pose[batch_number, 1 + 3] * point2[0] +\
                                 robot_body_pose[batch_number, 1 + 6] * point2[1] +\
                                 robot_body_pose[batch_number, 1 + 9] * point2[2] +\
                                 robot_body_pose[batch_number, 1],
                                 robot_body_pose[batch_number, 2 + 3] * point2[0] +\
                                 robot_body_pose[batch_number, 2 + 6] * point2[1] +\
                                 robot_body_pose[batch_number, 2 + 9] * point2[2] +\
                                 robot_body_pose[batch_number, 2])
    
    point3_transformed = wp.vec3(robot_body_pose[batch_number, 0 + 3] * point3[0] +\
                                 robot_body_pose[batch_number, 0 + 6] * point3[1] +\
                                 robot_body_pose[batch_number, 0 + 9] * point3[2] +\
                                 robot_body_pose[batch_number, 0],
                                 robot_body_pose[batch_number, 1 + 3] * point3[0] +\
                                 robot_body_pose[batch_number, 1 + 6] * point3[1] +\
                                 robot_body_pose[batch_number, 1 + 9] * point3[2] +\
                                 robot_body_pose[batch_number, 1],
                                 robot_body_pose[batch_number, 2 + 3] * point3[0] +\
                                 robot_body_pose[batch_number, 2 + 6] * point3[1] +\
                                 robot_body_pose[batch_number, 2 + 9] * point3[2] +\
                                 robot_body_pose[batch_number, 2])

    # Calculate the centroid of triangle points.
    center_point =\
        wp.div(wp.add(wp.add(point1_transformed, point2_transformed), point3_transformed), 3.)

    # Calculate normal of face from triangle points
    face_normal = wp.normalize(wp.cross(wp.sub(point2_transformed, point1_transformed),
                               wp.sub(point3_transformed, point1_transformed)))
    # Now rotate face normal
    face_normal_rot = rotate_normal(face_normal, ray_angles[ray_index, 0], ray_angles[ray_index, 1]) 

    # Area of the face using Heron's formula
    a = length(sub(point1_transformed, point2_transformed))
    b = length(sub(point1_transformed, point3_transformed))
    c = length(sub(point2_transformed, point3_transformed))
    # Half perimeter
    s = (a + b + c) / 2.
    # Arg for squaring root
    tri_arg = s * (s - a) * (s - b) * (s - c)
    # This case is invalid and means that the three lengths a,b,c can't actually
    # form a triangle. Not sure what's causing this. We just zero out such cases.
    if tri_arg < 0.:
        tri_arg = 0.
    face_area = pow(tri_arg, 0.5)

    # Now cast ray onto mesh object using this center face point and normal.
    d = float(0.0)              # hit distance along ray
    u = float(0.0)              # hit  face barycentric u
    v = float(0.0)              # hit  face barycentric u
    sign = float(0.0)           # hit face sign, value > 0 if ray hit front of face
    n = wp.vec3()               # hit face normal
    f = int(0)                  # hit face index
    inside = float(0.)          # < 0 if inside the mesh, 0 >= otherwise
    bary_u = float(0.)
    bary_v = float(0.)

    #ray_hit = wp.mesh_query_ray(object_mesh, center_point, face_normal_rot, max_depth, d, u, v, sign, n, f) 
    ray_hit = wp.mesh_query_point(object_mesh, center_point, max_depth, inside, f, bary_u, bary_v)
    closest_point = wp.vec3()
    if ray_hit:
        closest_point = mesh_eval_position(object_mesh, f, bary_u, bary_v)
    
    n = wp.normalize(closest_point - center_point)
    d = wp.length(closest_point - center_point)
    
    # Project the 3D body point velocity onto the distance direction to get the velocity along
    # direction.
    d_vel = wp.dot(center_point_vel, n)

    # Pull vertices of triangle of environmental mesh where the hit took place.
    tri_collision = int(0)
#    if face_area >= 0.:
#        mesh = wp.mesh_get(object_mesh)
#        world_point1 = mesh.points[mesh.indices[f, 0]] #wp.mesh_eval_position(mesh, f, 1.0, 0.0)
#        world_point2 = mesh.points[mesh.indices[f, 1]] #wp.mesh_eval_position(mesh, f, 0.0, 1.0)
#        world_point3 = mesh.points[mesh.indices[f, 2]] #wp.mesh_eval_position(mesh, f, 0.0, 0.0)
#
#        tri_collision = intersect_tri_tri(world_point1, world_point2, world_point3,
#                                          point1_transformed, point2_transformed, point3_transformed)

    # If the ray hits the front of the face, then this will be considered a valid distance
    offset = 1e-6
    d_min = 0.01 + offset # Creates a maximum response of 100 from barrier function.
    
    # Create signed distance. Positive is outside or 0, negative is inside.
    d = d * sign(inside)
    
    # Determining whether or not the query point is in a collision state
    collision_state[batch_number, face_index, ray_index] = 0

    if ray_hit:
        # We check if the query point is known to be inside the query mesh.
        # We also check to see if the robot triangle intersects the world triangle.
        # These two triangles are the ones associated with the nearest collision search.
        #if inside < 0. or d < 0.015: # or tri_collision > 0:
        #if inside < 0. or tri_collision > 0:
        if inside < 0. or d < .005:
            collision_state[batch_number, face_index, ray_index] = 1

    # Calculate fabric response given a valid returned signed distance.
    if ray_hit:
        # Clamp d so that you allow for up to some maximum response when d is small or negative.
        # We subtract a factor off of d to further pad the distance calculation.
        d_clamp = wp.clamp(d - 0.05, d_min, max_depth)

        # Set scaling on barrier type responses that factor in number of rays and face area
        # associated with ray.
        metric_scaling = 3. * face_area
        accel_scaling =  9.81 * 10.
        damping_scaling = float(0.0)
        if d_vel >= 0:
            damping_scaling = 0. #10.

        # Direction of priority is only along the normal, scaled by hit distance.
        metrics[batch_number, face_index, ray_index] = wp.mul(wp.outer(n, n), (metric_scaling)/(d_clamp-offset))

        # Force along the direction of the face normal, scaled by the hit distance
        # (force is on left side of equation, M xdd + f = 0)
        #forces[batch_number, face_index, ray_index] =\
#                wp.mul( face_normal_rot, accel_scaling/(d_clamp-offset)) + wp.mul( face_normal_rot, (damping_scaling * d_vel) /(d_clamp-offset))
        forces[batch_number, face_index, ray_index] =\
                n * (metric_scaling/(d_clamp-offset)) *\
                (accel_scaling * length_sq(center_point_vel) + damping_scaling * d_vel)
        #         (damping_scaling * d_vel) /(d_clamp-offset))

        # Calculate hit location
        ray_hit_points[batch_number, face_index * num_rays + ray_index] = wp.add(center_point, wp.mul(n, d))

    # If no valid ray hits, then zero out fabric term.
    else:
        # Set mass and acceleration to 0s if no valid hit.
        metrics[batch_number, face_index, ray_index] = wp.mul(metrics[batch_number, face_index, ray_index], 0.)
        forces[batch_number, face_index, ray_index] = wp.mul(forces[batch_number, face_index, ray_index], 0.)
        
        # Set hit location to 0s
        ray_hit_points[batch_number, face_index  * num_rays + ray_index] = wp.mul(n, 0.)

    # Fill out the linear map corresponding to the point used for ray casting.
    # X component
    linear_maps[batch_number, face_index, 0, 3] = center_point_body[0]
    linear_maps[batch_number, face_index, 1, 4] = center_point_body[0]
    linear_maps[batch_number, face_index, 2, 5] = center_point_body[0]
    
    # Y component
    linear_maps[batch_number, face_index, 0, 6] = center_point_body[1]
    linear_maps[batch_number, face_index, 1, 7] = center_point_body[1]
    linear_maps[batch_number, face_index, 2, 8] = center_point_body[1]
    
    # Z component
    linear_maps[batch_number, face_index, 0, 9] = center_point_body[2]
    linear_maps[batch_number, face_index, 1, 10] = center_point_body[2]
    linear_maps[batch_number, face_index, 2, 11] = center_point_body[2]
    
    # Origin
    linear_maps[batch_number, face_index, 0, 0] = 1.
    linear_maps[batch_number, face_index, 1, 1] = 1.
    linear_maps[batch_number, face_index, 2, 2] = 1.

def eval_repulsion_func(robot_body_pose, robot_body_velocity, allocated_data):

    num_threads = allocated_data['batch_size'] * allocated_data['num_faces']
    body_index = allocated_data['body_index']
    num_rays = allocated_data['ray_angles'].shape[0]

    wp.launch(kernel=eval_repulsion,
              dim=[num_threads, num_rays],
              inputs=[
                  robot_body_pose,
                  robot_body_velocity,
                  allocated_data['robot'].shape_geo_src[body_index].mesh.indices,
                  allocated_data['robot'].shape_geo_src[body_index].mesh.points,
                  allocated_data['object_id'],
                  allocated_data['max_distance'],
                  allocated_data['num_body_points'],
                  allocated_data['num_faces'],
                  allocated_data['ray_angles'],
                  num_rays,
                  allocated_data['collision_state']
                  ],
              outputs=[
                  allocated_data['metrics'],
                  allocated_data['forces'],
                  allocated_data['linear_maps'],
                  allocated_data['ray_hit_points']
                  ],
              device=allocated_data['device'])

# Define PyTorch autograd op to wrap repulsion kernel.
class Repulsion(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, robot_body_pose, robot_body_velocity, allocated_data):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.robot_body_pose = wp.torch.from_torch(robot_body_pose)
        ctx.robot_body_velocity = wp.torch.from_torch(robot_body_velocity)
        ctx.allocated_data = allocated_data
        
        with ctx.tape:
            eval_repulsion_func(
                ctx.robot_body_pose,
                ctx.robot_body_velocity,
                ctx.allocated_data)
        
        return (wp.torch.to_torch(ctx.allocated_data['metrics']),
                wp.torch.to_torch(ctx.allocated_data['forces']),
                wp.torch.to_torch(ctx.allocated_data['linear_maps']))

    @staticmethod
    def backward(ctx, adj_metrics, adj_forces, adj_linear_maps):

        # Map incoming Torch grads to our output variables
        # TODO: this should be allocated_data, not allocated_state
        grads = { ctx.allocated_state['metrics']:
                      wp.torch.from_torch(adj_metrics, dtype=mat33, ndim=3),
                  ctx.allocated_state['forces']:
                      wp.torch.from_torch(adj_forces, dtype=vec3, ndim=3),
                  ctx.allocated_state['linear_maps']:
                      wp.torch.from_torch(adj_linear_maps, dtype=float, ndim=4) }

        # Calculate gradients
        ctx.tape.backward(grads=grads)

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.robot_body_pose]),
                wp.torch.to_torch(ctx.tape.gradients[ctx.robot_body_velocity]),
                None)

class BodyRepulsion(BaseFabricTerm):
    """
    Implements repulsion between robot body and all known obstacles in the environment.
    """
    def __init__(self, is_forcing_policy, robot, batch_size, body_index, params, device):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        @param robot: robot container that holds mesh data for the robot.
        """
        super().__init__(is_forcing_policy, params, device)

        self.robot = robot
        max_distance = 1.

        # Generate random angles for perturbing rays within a cone.
        # These angles will be generated once and fixed using a space-filling
        # sampler. Generate random angles about x and y, so dimensionality will be 2
        dim_angles = 2
        # TODO: num_rays and cone angle should probably be settable from config file.
        num_rays = 10
        max_angle = 45. * (3.14 / 180.)
        # Create the generator
        sequencer = ghalton.GeneralizedHalton(dim_angles)
        # Sample the generator
        ray_angles = np.array(sequencer.get(num_rays))
        # Scale the result (which is currently between 0. and 1.) to (-max_angle, max_angle).
        ray_angles = 2. * max_angle * ray_angles - max_angle
        # Manually add in zero angles
        ray_angles = np.concatenate((ray_angles, np.zeros((1,2))), axis=0)
        num_rays += 1
        # NOTE: I am setting only to 1 ray for new distance method.
        ray_angles = np.zeros((1,2)) # only 
        num_rays = 1
        # Convert to warp array.
        ray_angles = wp.array(ray_angles, dtype=float, device=self.device)

        # Allocate various input and output arrays/data.
        num_body_points = len(robot.shape_geo_src[body_index].mesh.points)
        self.num_faces = int(len(robot.shape_geo_src[body_index].mesh.indices) / 3)
        self.num_rays = num_rays
        self.batch_size = batch_size
        self.allocated_data =\
            { 'robot': robot,
              'body_index': body_index,
              'batch_size': batch_size,
              'num_body_points': num_body_points,
              'num_faces': self.num_faces,
              'object_id': None,  # mesh id for object in scene
              'max_distance': max_distance,
              'ray_angles': ray_angles,
              'collision_state': wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=int),
              'metrics': wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=wp.mat33, device=self.device),
              'forces': wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=wp.vec3, device=self.device),
              'linear_maps': wp.zeros(shape=(batch_size, self.num_faces, 3, 12), dtype=float, device=self.device),
              'ray_hit_points': wp.zeros(shape=(batch_size, self.num_faces * self.num_rays), dtype=wp.vec3, device=self.device),
              'device': self.device }

        # The repulsion kernel will calculate both metrics and forces, but the two functions are separated
        # out below. So we will allocate for the force here and hold onto it and just return its value later.
        self.force = None

        # Origins of rays which are points on the mesh body
        self.mesh_points = None
        
        # Set initial collision state.
        self.in_collision = self.get_collision_status()

    def reallocate_variables(self, batch_size):
        self.batch_size = batch_size

        self.allocated_data['batch_size'] = batch_size
        self.allocated_data['collision_state'] = wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=int)
        self.allocated_data['metrics'] = wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=wp.mat33, device=self.device)
        self.allocated_data['forces'] = wp.zeros(shape=(batch_size, self.num_faces, self.num_rays), dtype=wp.vec3, device=self.device)
        self.allocated_data['linear_maps'] = wp.zeros(shape=(batch_size, self.num_faces, 3, 12), dtype=float, device=self.device)
        self.allocated_data['ray_hit_points'] = wp.zeros(shape=(batch_size, self.num_faces * self.num_rays), dtype=wp.vec3, device=self.device)

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return metric: policy metric
        """

        # First check if batch size has changed and if so, then reshape memory allocations
        if x.shape[0] != self.batch_size:
            self.reallocate_variables(x.shape[0])

        # TODO: need to update this to handle the mulitple objects. Currently just accessing the one object.
        #       Probably create an array if uint64's for kernel which are the object ids.
        #self.allocated_data['object_id'] = features['cylinder1']['object_mesh'].id

        if len(features) > 0:
            # Get first key name
            object_name = next(iter(features))
            self.allocated_data['object_id'] = features[object_name]['object_mesh'].id

            # Calculate 3x3 metric and acceleration response for all points on body across the batch.
            (metrics, forces, linear_maps) =\
                Repulsion.apply(x.float(),
                                xd.float(),
                                self.allocated_data)
            
            # Sum metrics and forces across cone rays.
            metrics = torch.sum(metrics, dim=2)
            forces = torch.sum(forces, dim=2)

            # Pull back metrics for every metric associated with every point in the batch and do a weighted
            # sum across all points for every instance in the batch.
            self.metric = torch.sum(linear_maps.transpose(2, 3) @ metrics @ linear_maps, dim=1)

            # Pull back forces
            self.force = torch.sum(linear_maps.transpose(2, 3) @ forces.unsqueeze(3), dim=1).squeeze(2)

            # NOTE: Currently not using this as a geometry, but we could. 
            # It adds more computational cost unless we figure out how to share
            # metric and force calculations for both forcing and nonforcing versions.
            # Currently commented out below.

            # If geometric policy, then make HD2
    #        if not self.is_forcing_policy:
    #            vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
    #            self.force = vel_squared * self.force

            # Generate robot mesh points where ray origins are located.
            self.mesh_points = (linear_maps @ x.unsqueeze(1).unsqueeze(3)).squeeze()
        else:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], device=self.device)

        return self.metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this repulsion term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: features (inputs) to pass to this term.
        @return force: policy force
        """

        # We already calculated the forces in the metric eval, so just return the value here.

        return self.force

    def get_body_points(self):
        """
        Returns the points used for ray casting on the robot mesh body.
        -----------------------------
        @return mesh_points: 3D point locations on the mesh
        """

        return self.mesh_points

    def get_ray_hit_points(self):
        """
        Returns the points on the object hit by rays.
        -----------------------------
        @return ray_hit_points: 3D point locations on the object.
        """

        return wp.torch.to_torch(self.allocated_data['ray_hit_points'])
        
    def get_collision_status(self):
        self.in_collision = wp.torch.to_torch(self.allocated_data['collision_state'])
        self.in_collision = torch.sum(torch.sum(self.in_collision, dim=2), dim=1)

        return self.in_collision
        
