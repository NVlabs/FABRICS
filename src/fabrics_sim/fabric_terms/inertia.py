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

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

class Inertia(BaseFabricTerm):
    """
    Implements a fabric attractor term.
    """
    def __init__(self, is_forcing_policy, params, device):
        """
        Constructor.
        -----------------------------
        """
        super().__init__(is_forcing_policy, params, device)

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return M: metric
        """
        # Check to see if no target is set. If true, then
        # set priority metric/mass to zeros as this will drop
        # out the effect of this fabric component entirely.

        metric_collapsed = self.params['isotropic_mass'] * torch.ones_like(x)
        self.metric = torch.diag_embed(metric_collapsed)

        return self.metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: features (inputs) to pass to this term.
        @return force: batch bxn tensor of policy forces
        """
        force = torch.zeros(x.shape, requires_grad=True, device='cuda')

        return force
