# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a joint upper limit map.
"""

import os
import torch
from fabrics_sim.taskmaps.maps_base import BaseMap

class UpperJointLimitMap(BaseMap):
    def __init__(self, upper_joint_limits, batch_size, device):
        """
        Initializes the upper joint limit task map.
        -----------------------------
        :param upper_joint_limits: a list of joint limits
        :param batch_size: int, size of batch
        :param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        self.upper_joint_limits = upper_joint_limits
        self.upper_joint_limits_batch = None
        self.batch_size = batch_size

        self.init_limits()

    def init_limits(self):
        if self.upper_joint_limits_batch is None:
            num_joints = len(self.upper_joint_limits)
            self.upper_joint_limits_batch =\
                torch.zeros(self.batch_size, num_joints, device=self.device)
            with torch.no_grad():
                for i in range(num_joints):
                    self.upper_joint_limits_batch[:,i] = self.upper_joint_limits[i]

                # Create the Jacobian
                dim = num_joints
                single_jacobian = -torch.eye(dim, dim, device=self.device)
                single_jacobian = single_jacobian.reshape((1, dim, dim))
                self.jacobian = single_jacobian.repeat(self.batch_size, 1, 1)

    def forward_position(self, q, features):
        x = self.upper_joint_limits_batch - q

        return (x, self.jacobian)
