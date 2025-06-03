# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import time
from abc import ABC, abstractmethod
from fabrics_sim.energy.energy import Energy

class EuclideanEnergy(Energy):
    """
    Implements Euclidean energy.
    """
    def __init__(self, batch_size, num_joints, device):
        """
        Constructor that allocates for energy, metric, and xdd
        so that it can be calculated once and recalled subsequently.
        """
        super().__init__(device)

        self.init = False
        self.metric = None
        self.force = None
        self.batch_size = batch_size
        self.num_joints = num_joints

        self.init_energy()

    def init_energy(self):
        # If the energy, metric, and xdd (acceleration) has not yet been calculated,
        # then calculated once and reuse subsequently.
        # Rebuild all static tensors if their batch size does not match that of the
        # incoming state.
        if self.init is False or self.metric.shape[0] != x.shape[0]:
            #metric_collapsed = torch.ones_like(x)
            metric_collapsed = torch.ones(self.batch_size, self.num_joints, device=self.device)
            self.metric = torch.diag_embed(metric_collapsed)

            # Since mass is constant, there is no curvature force from Euler-Lagrange
            # equation. No force means no acceleration.
            self.force = torch.zeros(self.batch_size, self.num_joints, requires_grad=False, device=self.device)

            self.init = True

    def energy_eval(self, x, xd):
        """
        Evaluates the Euclidean energy, its metric, and corresponding acceleration.
        ------------------------------------------
        @param x: batch position, a bxm tensor 
        @param xd: batch velocity, a bxm tensor 
        @return self.metric: batch metric, a bxmxm tensor
        @return self.force: batch force, a bxm tensor
        @return self.energy: batch energy, a b-sized tensor
        """

        # Calculate energy. The energy is L = 1/2 xd' I xd
        energy = 0.5 * torch.sum(xd*xd, dim=1).unsqueeze(1)

        return (self.metric, self.force, energy)
