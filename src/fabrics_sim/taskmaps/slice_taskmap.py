# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os
import torch
import time
from fabrics_sim.taskmaps.maps_base import BaseMap

class SliceMap(BaseMap):
    """
    Implements subselecting dimensions of the input map.
    """
    def __init__(self, indices, device):
        """
        Constructor.
        ------------------------------------------
        @param indices: list of indices that will be sliced out to construct new map
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        # Allocate for the map's Jacobian so it can be set once and recalled.
        self.indices = torch.tensor(indices, device=self.device)
        self.jacobian = None

    def forward_position(self, q, features):
        """
        Evaluates the identity map and its Jacobian given the input position.
        ------------------------------------------
        @param q: batch position, a bxn tensor 
        @return q: batch position, a bxn tensor 
        @return self.jacobian: batch jacobian, a bxnxn tensor
        """
        # If Jacobian (which is fixed) has not yet been calculated, then calculate once
        # and save result for subsequent calls.
        # If batch size of Jacobian does not equal that of incoming state, then rebuild
        # the Jacobian.
        if self.jacobian is None or self.jacobian.shape[0] != q.shape[0]:
            num_cols = q.shape[1]
            num_rows = len(self.indices)
            single_jacobian = torch.zeros(num_rows, num_cols, device=self.device)
            single_jacobian[range(num_rows), self.indices] = 1.
            single_jacobian = single_jacobian.reshape((1, num_rows, num_cols))
            self.jacobian = single_jacobian.repeat(q.shape[0], 1, 1)

        # Incoming position is immediately mapped to output position because this map
        # is the identity.

        x = q[:, self.indices]

        return (x, self.jacobian)
