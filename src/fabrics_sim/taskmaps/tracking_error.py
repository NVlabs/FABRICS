# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os
import torch
from fabrics_sim.taskmaps.maps_base import BaseMap

class TrackingError(BaseMap):
    """
    Implements a tracking error map with the intended goal to keep the values
    in this map positive.
    """
    def __init__(self, joint_error_limits, device):
        """
        @param joint_error_limits: a list of the maximum positive joint position errors
        """
        super().__init__(device)

        self.joint_error_limits = torch.tensor(joint_error_limits, device='cuda')
        self.joint_error_limits_batch = None

    def forward_position(self, q, features):
        if features is None:
            raise ValueError('Need to pass measured joint positions in as features.')

        if (self.joint_error_limits_batch is None) or \
           (self.joint_error_limits_batch.shape[0] != q.shape[0]):
            self.joint_error_limits_batch = torch.zeros(q.shape, device='cuda')
            with torch.no_grad():
                for i in range(q.shape[1]):
                    self.joint_error_limits_batch[:,i] = self.joint_error_limits[i]
        
        # Features here are the current measured joint positions
        error = 10. * (q - features)
        x = self.joint_error_limits_batch - error.abs()

        # TODO: need to ensure that below is actually element-wise division
        # TODO: need to check if this jacobian is accurate
        jacobian = torch.diag_embed(-error / (error.abs() + 1e-5))

        return (x, jacobian)
