# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a base fabrics component class.
"""

import torch
from abc import ABC, abstractmethod

class BaseFabricTerm(torch.nn.Module):
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        """
        Initializes the base fabric class.
        -----------------------------
        """
        super().__init__()

        self.device = device
        self.params = params
        self.graph_capturable = graph_capturable
        # Immediately enable this fabric term by toggling to 1.
        self.params["toggle"] = 1.
        self._is_forcing_policy = is_forcing_policy

        # Allocate for metric that can be used to calculate force
        self.metric = None
        self.force = None

    @property
    def is_forcing_policy(self):
        return self._is_forcing_policy

    def set_param_value(self, param_name, param_value):
        """
        Sets a new value for a particular parameter.
        -----------------------------
        :param param_name: str, name of the parameter
        :param xd: float(s) or bool, value(s) of the parameter
        """

        self.params[param_name] = param_value

    @abstractmethod
    def metric_eval(self, x, xd, features):
        """
        Calculates the metric (aka mass) based on state.
        -----------------------------
        :param x: leaf space position
        :param xd: leaf space velocity
        :return metric: policy metric
        """

    @abstractmethod
    def force_eval(self, x, xd, features):
        """
        Calculates the policy force based on state.
        -----------------------------
        :param x: leaf space position
        :param xd: leaf space velocity
        :return force: leaf space force
        """

    def forward(self, x, xd, features):
        """
        Calculates the fabric metric and force.
        -----------------------------
        :param x: leaf space position
        :param xd: leaf space velocity
        :return M: leaf space metric
        :return force: leaf space force
        """

        self.metric_eval(x, xd, features)
        self.force_eval(x, xd, features)

        return [self.metric, self.force]
