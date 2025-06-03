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

class Attractor_1D(BaseFabricTerm):
    """
    Implements a fabric attractor term.
    """
    def __init__(self, is_forcing_policy, params, device):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device)

        # Allocating for a few signals so we can calculate once and reuse
        # in the forward pass.
        self.position_error = None
        self.position_error_dir = None
        self.position_error_dir_matrix = None

    # TODO: pass features.
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

        if features is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=True, device=self.device)

        # TODO: this is creating mass. Turn this into a function after we test this basic version first.
        else:
            self.position_error = features - x
            self.position_error_dir = torch.nn.functional.normalize(self.position_error)
            self.position_error_dir_matrix = self.position_error_dir.unsqueeze(2) @ \
                                             self.position_error_dir.unsqueeze(1)
            self.metric = self.params['max_mass'] * self.position_error_dir_matrix

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
        # Check to see if no target is set. If true, then
        # set acceleration to zeros.
        xdd = None
        if features is None:
            xdd = torch.zeros(x.shape, requires_grad=True, device=self.device)
        else:
            scaling = self.params['conical_gain'] *\
                      torch.tanh(self.params['conical_sharpness'] *\
                        torch.linalg.norm(self.position_error, dim=1))
            xdd_not_hd2 = scaling.unsqueeze(1) * self.position_error_dir

            # Add on damping when sufficiently close to target.
            # TODO: Make an HD2 version by squaring velocity
            if not self.is_forcing_policy:
                vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
                xdd = vel_squared * xdd_not_hd2
            else:
                damping = (torch.linalg.norm(self.position_error, dim=1) <
                            self.params['damping_radius'])
                damping = self.params['damping'] * damping

                #xdd = xdd_not_hd2 - damping.unsqueeze(1) *\
                #        (self.position_error_dir_matrix @ xd.unsqueeze(2)).squeeze()
                xdd = xdd_not_hd2 - damping.unsqueeze(1) * xd

        # Convert to force.
        force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

        return force
        
