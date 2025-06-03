# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements an inertia fabric components
"""

import os

import torch
import yaml

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

class CspaceAttractor(BaseFabricTerm):
    """
    Creates a cspace attractor term.
    """
    def __init__(self, is_forcing_policy, params, device):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device)

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return metric: policy metric
        """

        metric_collapsed = self.params['isotropic_metric'] * torch.ones_like(x)
        self.metric = torch.diag_embed(metric_collapsed)

        return self.metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return force: batch bxn tensor of policy forces
        """
        
        vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
        if x.shape != features.shape:
            raise ValueError('Dimensionality mismatch between position and position target in \
                   cspace attractor.')
        position_error = x - features

        scaling = self.params['conical_gain'] *\
                  torch.tanh(self.params['conical_sharpness'] *\
                    torch.linalg.norm(position_error, dim=1))

        xdd_not_hd2 = -scaling.unsqueeze(1) * torch.nn.functional.normalize(position_error)
        
        xdd = vel_squared * xdd_not_hd2
        
        # Convert to force.
        force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

        return force

