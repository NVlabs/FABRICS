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

class LinearMap(BaseMap):
    """
    Implements a linear map. This coule be derived from something
    like PCA on motion data.
    """
    def __init__(self, linear_map, device):
        """
        Constructor that sets the linear map.
        ------------------------------------------
        @param linear_map: mxn tensor that contitutes the linear map
                           m is the dimension of the mapped space
                           n in the dimension of the input space
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        # Set the linear map to be recalled later
        self.linear_map = linear_map

        # We assume the map is fixed, so we deactivate its gradient.
        self.linear_map.requires_grad = False

        # Allocate for the map's Jacobian so it can be set once and recalled.
        self.jacobian = None


    def forward_position(self, q, features):
        """
        Evaluates the linear map given the input position.
        ------------------------------------------
        @param q: batch position, a bxn tensor 
        """
        # Evaluate the linear map position.
        x = q @ torch.transpose(self.linear_map, 0, 1)
        
        # If Jacobian (which is fixed) has not yet been calculated, then calculate once
        # and save result for subsequent calls.
        # If batch size of Jacobian does not equal that of incoming state, then rebuild
        # the Jacobian.
        if self.jacobian is None or self.jacobian.shape[0] != q.shape[0]:
            # Calculate its Jacobian (which is fixed)
            single_jacobian = self.linear_map
            single_jacobian =\
                single_jacobian.reshape((1, self.linear_map.shape[0], self.linear_map.shape[1]))
            self.jacobian = single_jacobian.repeat(q.shape[0], 1, 1)

        return (x, self.jacobian)
