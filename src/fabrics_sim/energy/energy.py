# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Implements a base energy class.
"""

import torch
from abc import ABC, abstractmethod

class Energy(torch.nn.Module):
    def __init__(self, device):
        """
        Initializes the base energy class.
        -----------------------------
        """

        super().__init__()

        self.device = device
        self.params = None

    @abstractmethod
    def energy_eval(self, x, xd):
        """
        Calculates the energy metric, acceleration, and energy scalar
        as a function of state.
        -----------------------------
        :param x: leaf space position
        :param xd: leaf space velocity
        :return M: metric
        :return force: force
        :return energy: scalar energy in this space
        """

    def forward(self, x, xd):

        return self.energy_eval(x, xd)

