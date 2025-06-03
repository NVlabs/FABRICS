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

class JointSpeedLimitRepulsion(BaseFabricTerm):
    """
    Implements a fabric joint speed limit repulsion term.
    """
    def __init__(self, is_forcing_policy, params, device):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device)

        self.joint_speed_limit =\
                torch.tensor(self.params['velocity_limits'], device=self.device)
        self._kEpsilon = 1e-6

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this repulsion term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return M: metric
        """

        # Velocity distance to limit.
        d_vel = torch.abs(xd) - self.joint_speed_limit

        # Diagonal metric of barrier terms as functions of distance to velocity limit.
        metric_diag = torch.clamp(self.params['metric_scalar']/(d_vel * d_vel + self._kEpsilon),
                                 max=self.params['max_metric'])
        self.metric = torch.diag_embed(metric_diag)

        return self.metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this repulsion term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: features (inputs) to pass to this term.
        @return force: batch bxn tensor of policy forces
        """
        
        # Velocity distance to upper joint velocity limit.
        d_vel_upper = torch.clamp(xd,
                                  min=-self.joint_speed_limit,
                                  max=self.joint_speed_limit) -\
                                          self.joint_speed_limit
        
        # Velocity distance to lower joint velocity limit.
        d_vel_lower = torch.clamp(xd,
                                  min=-self.joint_speed_limit,
                                  max=self.joint_speed_limit) +\
                                          self.joint_speed_limit

        # Calculate acceleration contribution from barrier function looking at distance
        # to upper velocity limit and distance to lower velocity limit. If distance to
        # upper velocity limit is decreasing, then accelerate negatively away. If distance
        # to lower velocity limit is decreasing, then accelerate positively away.
        xdd_barrier_unclamped =\
            -self.params['barrier_gain'] / (d_vel_upper * d_vel_upper + self._kEpsilon) +\
            self.params['barrier_gain'] / (d_vel_lower * d_vel_lower + self._kEpsilon)

        # Clamp acceleration to be within set limits.
        xdd = torch.clamp(xdd_barrier_unclamped,
                                  min=-self.params['barrier_max_acceleration'],
                                  max=self.params['barrier_max_acceleration'])

        # Convert to force.
        force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze()

        return force
