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
import time

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

class JointLimitRepulsion(BaseFabricTerm):
    """
    Implements a fabric joint limit repulsion term.
    """
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)

        self._kEpsilon = 1e-6
        self.params['metric_scalar'] = torch.tensor(self.params['metric_scalar'], device=device)
        self.params['max_metric'] = torch.tensor(self.params['max_metric'], device=device)
        self._min_x_delta = self.compute_min_x_delta()

        self.ones_like_x = None

    def compute_min_x_delta(self):
        return torch.tensor((self.params['metric_scalar'] / self.params['max_metric'])**0.5,
                            device=self.device)

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return metric: policy metric
        """
        if self.metric is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.ones_like_x = torch.ones_like(x)
        
        if self.graph_capturable:
            self.metric.zero_().detach_()
            self.force.zero_().detach_()
        else:
            self.metric = torch.zeros_like(self.metric)
            self.force = torch.zeros_like(self.force)

        x_delta = torch.maximum(self._min_x_delta, x - self.params['metric_exploder_offset'])

        if self.params['velocity_gate']:
            # Trigger set to slightly positive position and velocity which helps with numerical
            # issues of this gate turning on/off with large dt. The effect of that is that you can
            # creep/oscillate through the joint limit.
            activation_trigger = torch.logical_or(xd < self.params['breakaway_velocity'],
                                                  x < self.params['breakaway_distance'])
            if self.graph_capturable:
                self.metric.copy_(torch.diag_embed(activation_trigger * self.params['metric_scalar'] / x_delta ** 2))
            else:
                self.metric =\
                    torch.diag_embed(activation_trigger * self.params['metric_scalar'] / x_delta ** 2)
        else:
            if self.graph_capturable:
                self.metric.copy_(torch.diag_embed(self.params['metric_scalar'] / x_delta ** 2))
            else:
                self.metric = torch.diag_embed(self.params['metric_scalar'] / x_delta ** 2)

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this repulsion term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: features (inputs) to pass to this term.
        @return force: batch bxn tensor of policy forces
        """

        if not self.is_forcing_policy:
            vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
            xdd = vel_squared * self.params['soft_relu_gain'] * self.ones_like_x
        else:
            # If velocity is negative (motion towards limit), then engage damping
            damping_gain = (xd <= 0) * self.params['damping_gain']
            xdd = self.params['soft_relu_gain'] * self.ones_like_x - damping_gain  * xd
        
        # Convert to force.
        if self.graph_capturable:
            self.force.copy_(-torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2))
        else:
            self.force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

