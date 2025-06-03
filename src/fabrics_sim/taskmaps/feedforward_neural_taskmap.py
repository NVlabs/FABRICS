# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os
import torch
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import yaml

from fabrics_sim.taskmaps.maps_base import BaseMap
from fabrics_sim.utils.path_utils import get_params_path 
from fabrics_sim.utils.rff import RFF

class FeedforwardNeuralMap(BaseMap):
    """
    Implements a neural network taskmap.
    """
    def __init__(self, parameters, device):
        """
        Constructor that establishes the layers for this taskmap.
        ------------------------------------------
        @param parameters: dict, of paramters for constructing the NN task map.
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        # Get parameters for this taskmap.
        self.params = parameters

        # Setting up some dimensions
        num_hidden_units1 = self.params['num_hidden_units1']
        num_hidden_units2 = self.params['num_hidden_units2']
        num_hidden_units3 = self.params['num_hidden_units3']
        num_hidden_units4 = self.params['num_hidden_units4']
        root_dim = self.params['root_dim']
        space_dim = self.params['space_dim']
            
        # Settings for Fourier features.
        num_fourier_features = 128
        num_fourier_inputs = 1 * num_fourier_features
        spectral_scaling = 2.
        kernel_type = 'RBF'
        self.rff_features = RFF(num_fourier_features,
                                root_dim,
                                spectral_scaling,
                                kernel=kernel_type,
                                device=self.device)

        # Inputs will be position.
        self.taskmap_fc1 = spectral_norm(torch.nn.Linear(num_fourier_features, num_hidden_units1, device=self.device))
        self.taskmap_fc2 = spectral_norm(torch.nn.Linear(num_hidden_units1, num_hidden_units2, device=self.device))
        self.taskmap_fc3 = spectral_norm(torch.nn.Linear(num_hidden_units2, num_hidden_units3, device=self.device))
        self.taskmap_fc4 = spectral_norm(torch.nn.Linear(num_hidden_units3, num_hidden_units4, device=self.device))
        self.taskmap_fc5 = spectral_norm(torch.nn.Linear(num_hidden_units4, space_dim, device=self.device))

        # Change layer initializations.
        sigma = 1e-2
        torch.nn.init.normal_(self.taskmap_fc1.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc1.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc2.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc2.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc3.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc3.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc4.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc4.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc5.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.taskmap_fc5.bias, mean=0., std=sigma)
        
        self.elu_alpha = 5.

    def forward_position(self, q, features):
        """
        Evaluates the neural map given the input position.
        ------------------------------------------
        @param q: batch position, a bxn tensor
        """

        # Calculate the forward map by passing the input position through the neural network.
        rff_hid = self.rff_features.to_features(q)
        hid = self.taskmap_fc2(F.elu(self.taskmap_fc1(rff_hid), alpha=self.elu_alpha))
        hid2 = self.taskmap_fc3(F.elu(hid, alpha=self.elu_alpha))
        hid3 = self.taskmap_fc4(F.elu(hid2, alpha=self.elu_alpha))
        x = self.taskmap_fc5(F.elu(hid3, alpha=self.elu_alpha))
        # TODO: Add selection of hyperspherical map and then enable the following two lines and the one
        # in the forward function.
        #radius = .5
        #x = radius * torch.nn.functional.normalize(x)

        # Function to facilitate the calculation of this map's Jacobian.
        taskmap_coordinate_index = 0
        def forward_function(q):
            rff_hid = self.rff_features.to_features(q)
            hid = self.taskmap_fc2(F.elu(self.taskmap_fc1(rff_hid), alpha=self.elu_alpha))
            hid2 = self.taskmap_fc3(F.elu(hid, alpha=self.elu_alpha))
            hid3 = self.taskmap_fc4(F.elu(hid2, alpha=self.elu_alpha))
            x = self.taskmap_fc5(F.elu(hid3, alpha=self.elu_alpha))
            #x = radius * torch.nn.functional.normalize(x)
            ones = torch.ones(x.shape[0], device=self.device)
            summed_map = (ones * x[:,taskmap_coordinate_index]).squeeze().sum(dim=0)

            return summed_map

        taskmap_grads = []
        for i in range(x.shape[1]):
            taskmap_grad = torch.autograd.functional.jacobian(forward_function, q, create_graph=True)
            taskmap_coordinate_index += 1
            taskmap_grads.append(taskmap_grad.unsqueeze(1))

        jacobian = torch.cat(taskmap_grads, dim=1)

        return (x, jacobian)
