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

class Attractor(BaseFabricTerm):
    """
    Implements a fabric attractor term.
    """
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)
        
        # Allocating for a few signals so we can calculate once and reuse
        # in the forward pass.
        self.position_error = None
        self._damping_position = None

        # Turn some gains into tensors
        is_list = isinstance(self.params['min_isotropic_mass'], list)
        if is_list:
            self.params['min_isotropic_mass'] = torch.tensor(self.params['min_isotropic_mass'],
                                                             device=device)
            self.params['max_isotropic_mass'] = torch.tensor(self.params['max_isotropic_mass'],
                                                             device=device)
            self.params['conical_gain'] = torch.tensor(self.params['conical_gain'],
                                                             device=device)

            if is_forcing_policy:
                self.params['damping'] = torch.tensor(self.params['damping'],
                                                      device=device)
                
        self.ones_like_x = None

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

        if features is not None:
            self.position_error = x - features
            scaled_mass =\
                    (self.params['max_isotropic_mass'] - self.params['min_isotropic_mass']) *\
                    (.5 * torch.tanh(-self.params['mass_sharpness'] * \
                        (torch.linalg.norm(self.position_error, dim=1) -\
                         self.params['mass_switch_offset'])) + .5).unsqueeze(1) +\
                    self.params['min_isotropic_mass']
            metric_collapsed = scaled_mass * self.ones_like_x # torch.ones_like(x)
            if self.graph_capturable:
                self.metric.copy_(torch.diag_embed(metric_collapsed))
            else:
                self.metric = torch.diag_embed(metric_collapsed)

        if torch.is_tensor(self.params['toggle']):
            if self.graph_capturable:
                self.metric.copy_(self.params['toggle'].unsqueeze(2) * self.metric)
            else:
                self.metric = self.params['toggle'].unsqueeze(2) * self.metric
        else:
            if self.graph_capturable:
                self.metric.copy_(self.params['toggle'] * self.metric)
            else:
                self.metric = self.params['toggle'] * self.metric

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
        #xdd = None
        if features is None:
            xdd = torch.zeros(x.shape, device=self.device)
        else:
            if x.shape != features.shape:
                raise ValueError('Dimensionality mismatch between position and position target in \
                       attractor.')

            position_error = x - features
            scaling = self.params['conical_gain'] *\
                      torch.tanh(self.params['conical_sharpness'] *\
                        torch.linalg.norm(position_error, dim=1)).unsqueeze(1)

            xdd_not_hd2 = -scaling * torch.nn.functional.normalize(position_error)

            # If not forcing policy, convert into geometry.
            if not self.is_forcing_policy:
                vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
                xdd = vel_squared * xdd_not_hd2
            else:
                # Default sets damping position equal to pose target.
                if self._damping_position is None:
                    self._damping_position = features
            
                # If position shape doesn't align with target shape, then throw error.
                if x.shape != self._damping_position.shape:
                    print('pos shape', x.shape)
                    print('damping shape', self._damping_position.shape)
                    raise ValueError('Dimensionality mismatch between position and damping target in \
                           attractor.')

                # Add on damping when sufficiently close to damping position.
                damping_position_error = x - self._damping_position
                damping = (.5 * torch.tanh(-self.params['damping_sharpness'] * \
                        (torch.linalg.norm(damping_position_error, dim=1) -\
                         self.params['damping_radius'])).unsqueeze(1) + .5)
                damping = self.params['damping'] * damping

                xdd = xdd_not_hd2 - damping * xd

        # Convert to force.
        if self.graph_capturable:
            self.force.copy_(-torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2))
        else:
            self.force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

        # Finally, reset damping position to None to force that
        # by default: the damping position will be aligned with the position target the next cycle
        # NOTE: damping position can also be set externally and placed at a different location
        # from the target position via the setter damping position attribute below.
        self._damping_position = None

    @property
    def damping_position(self):
        return self._damping_position

    @damping_position.setter
    def damping_position(self, damping_position):
        self._damping_position = damping_position
