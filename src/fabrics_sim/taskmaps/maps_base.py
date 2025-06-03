# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a base map class.
"""

import torch
from abc import ABC, abstractmethod

class BaseMap(torch.nn.Module):
    def __init__(self, device):
        """
        Initializes the base map class.
        -----------------------------
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__()
        self.device = device

    @abstractmethod
    def forward_position(self, q, features):
        """
        Calculates the leaf space position
        -----------------------------
        :param q: root position
        :return x: leaf position
        """

    def forward(self, q, features):
        return self.forward_position(q, features)




