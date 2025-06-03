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

import warp as wp
import warp.torch

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.taskmaps.maps_base import BaseMap

# Define PyTorch autograd op to wrap foward kinematics
# function.
class RobotKinematics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, joint_q, robot_kinematics):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.joint_q = wp.torch.from_torch(joint_q)
        ctx.robot_kinematics = robot_kinematics
        
        with ctx.tape:
            ctx.robot_kinematics.eval(ctx.joint_q, jacobians=True)
            #ctx.robot_kinematics.eval(ctx.joint_q, batch_qd=ctx.joint_q, velocities=True, jacobians=True)
        
        return (wp.torch.to_torch(ctx.robot_kinematics.batch_link_transforms),
                wp.torch.to_torch(ctx.robot_kinematics.batch_link_jacobians))

    @staticmethod
    def backward(ctx, adj_link_transforms, adj_jacobians):

        # Map incoming Torch grads to our output variables
        grads = { ctx.robot_kinematics.batch_link_transforms:
                      wp.torch.from_torch(adj_link_transforms, dtype=wp.transform),
                  ctx.robot_kinematics.batch_link_jacobians:
                      wp.torch.from_torch(adj_jacobians, dtype=wp.vec3) }

        # Calculate gradients
        ctx.tape.zero()
        ctx.tape.backward(grads=grads)

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.joint_q]),
                None,
                None)

class RobotFrameOriginsTaskMap(BaseMap):
    def __init__(self, urdf_path, link_names, batch_size, device):
        """
        Constructor for building the desired robot taskmap.
        -----------------------------------------
        :param urdf_path: str, robot URDF filepath
        :param link_names: list of link names (str) of the robot to build the taskmap
        :param batch_size: int, size of the batch of robots
        :param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        # Allocate for robot kinemtics, the relevant link indices, and the batch size.
        self.urdf_path = urdf_path
        self.robot_kinematics = None
        self.link_names = link_names
        self.link_indices = None
        self.batch_size = batch_size

        self.init_robot_kinematics(self.batch_size)

    def init_robot_kinematics(self, batch_size):
        # Create the robot kinematics object that wraps several Warp kernels for computing
        # forward kinematics
        multithreading = False
        self.robot_kinematics = Kinematics(self.urdf_path, batch_size, multithreading,
                                           device=self.device)

        self.link_indices =  []
        for link_name in self.link_names:
            self.link_indices.append(self.robot_kinematics.get_link_index(link_name))
        self.link_indices = torch.tensor(self.link_indices, device=self.device)

        self.batch_size = batch_size

    def forward_position(self, q, features):
        # Check if the batch size matches the batch size of the incoming q. If not,
        # then re-initialize the robots kinematics.
#        if self.batch_size != q.shape[0]:
#            self.init_robot_kinematics(q.shape[0])

        # Calculate the link transforms and their origin Jacobians.
        link_transforms, jacobians = RobotKinematics.apply(q, self.robot_kinematics)

        # Pull out the position of the origins and stack them across all desired frames.
        x = link_transforms[:, self.link_indices, :3].reshape((self.batch_size,
                                                               len(self.link_indices) * 3))

        # Pull out the Jacobians and stack them for the desired frames.
        # jacobian is of shape (batch_size, num_links, root_dim, 3)
        # so we transpose the last two dimensions to get a
        # jacobian of shape (batch_size, num_links, 3, root_dim)
        # and then reshape it to (batch_size, num_links * 3, root_dim)
        jacobian = jacobians[:, self.link_indices, :, :].transpose(2,3).reshape(
                        self.batch_size, len(self.link_indices) * 3, q.shape[1])

        return (x, jacobian)




