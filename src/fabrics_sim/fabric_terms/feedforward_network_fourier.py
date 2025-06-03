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
import yaml
import time

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm
from fabrics_sim.utils.utils import jacobian, check_numerical_jacobian
from fabrics_sim.utils.rff import RFF

class FeedforwardFourierTerm(BaseFabricTerm):
    """
    Creates a neural feedforward fabric term with Random
    Fourier Features applied on the inputs.
    """
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        """
        Constructor. This will set up the network layers of this fabric term.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)

        # Setting up some dimensions for the MLP.
        num_hidden_units = self.params['num_hidden_units']

        # Dimensionality of the space this term lives in.
        space_dim = self.params['space_dim']
        self.space_dim = space_dim

        # Settings for Fourier features.
        # TODO: need to make these settable parameters.
        num_fourier_features = 1000
        num_fourier_inputs = 1 * num_fourier_features
        spectral_scaling = 5.
        kernel_type = 'Matern12'
        
        # Layer initialization sigma
        sigma = 1e-2

        # Setting up layers for acceleration policy and metric
        # These are fully connected layers.
        num_inputs_policy = None
        num_inputs_metric = None
        if not self.is_forcing_policy:
            # Inputs will be position, velocity normalized, and the additional feature.
            num_inputs_policy = self.params['dim_additional_features'] + 2 * space_dim
        
            # Setting up layers for metric.
            # Inputs will be position, velocity normalized, and any additional features.
            num_inputs_metric = self.params['dim_additional_features'] + 2 * space_dim
        else:
            # If a forcing policy, then the NN can only be a function of position
            # and some additional features like a target.
            num_inputs_policy = self.params['dim_additional_features'] + space_dim
           
            # Setting up layers for metric.
            # Inputs will be position, velocity normalize, and any additional features.
            num_inputs_metric = self.params['dim_additional_features'] + space_dim
        
        # Setting up layers for policy
        self.policy_rff = RFF(num_fourier_features,
                              num_inputs_policy,
                              spectral_scaling,
                              kernel=kernel_type,
                              device=self.device)
        self.policy_fc1 = torch.nn.Linear(num_fourier_inputs, num_hidden_units,
                                          device=self.device)
        self.policy_fc2 = torch.nn.Linear(num_hidden_units, num_hidden_units,
                                          device=self.device)
        self.policy_fc3 = torch.nn.Linear(num_hidden_units, space_dim, device=self.device)
        
        # Change layer initializations.
        torch.nn.init.normal_(self.policy_fc1.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.policy_fc1.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.policy_fc2.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.policy_fc2.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.policy_fc3.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.policy_fc3.bias, mean=0., std=sigma)

        # Setting up layers for metric.
        self.metric_rff = RFF(num_fourier_features, num_inputs_metric, spectral_scaling, kernel=kernel_type, device=self.device)
        self.metric_fc1 = torch.nn.Linear(num_fourier_inputs, num_hidden_units, device=self.device)
        self.metric_fc2 = torch.nn.Linear(num_hidden_units, num_hidden_units, device=self.device)
        # Output are the Cholesky factors
        num_outputs = sum(range(1, space_dim + 1))
        self.metric_fc3 = torch.nn.Linear(num_hidden_units, num_outputs, device=self.device)

        # Change layer initializations.
        torch.nn.init.normal_(self.metric_fc1.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.metric_fc1.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.metric_fc2.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.metric_fc2.bias, mean=0., std=sigma)
        torch.nn.init.normal_(self.metric_fc3.weight, mean=0., std=sigma)
        torch.nn.init.normal_(self.metric_fc3.bias, mean=0., std=sigma)

        if self.is_forcing_policy:
            # Setting up layers for positive damping scalar.
            # Damping can be a function of both position and velocity of the space
            # and additional features.
            num_inputs_damping = self.params['dim_additional_features'] + 2 * space_dim
            self.damping_rff = RFF(num_fourier_features, num_inputs_damping, spectral_scaling, kernel=kernel_type, device=self.device)
            self.damping_fc1 = torch.nn.Linear(num_fourier_inputs, num_hidden_units, device=self.device)
            self.damping_fc2 = torch.nn.Linear(num_hidden_units, num_hidden_units, device=self.device)
            self.damping_fc3 = torch.nn.Linear(num_hidden_units, 1, device=self.device)
        
            # Change layer initializations.
            torch.nn.init.normal_(self.damping_fc1.weight, mean=0., std=sigma)
            torch.nn.init.normal_(self.damping_fc1.bias, mean=0., std=sigma)
            torch.nn.init.normal_(self.damping_fc2.weight, mean=0., std=sigma)
            torch.nn.init.normal_(self.damping_fc2.bias, mean=0., std=sigma)
            torch.nn.init.normal_(self.damping_fc3.weight, mean=0., std=sigma)
            torch.nn.init.normal_(self.damping_fc3.bias, mean=0., std=sigma)

        # We will use ELUs as the activation function. Dialing down
        # the alpha parameter for this function improves gradient flow.
        self.elu_alpha = 0.05

        # Allocate for this because we will need it in deriving a potential-based acceleration.
        self.metric = None
        self.metric_small_diag = None
        self.xdd = None

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: batch bxn tensor of positions
        @param xd: batch bxn tensor of velocities
        @param features: batch bxm tensor of additional features to this fabric term.
        @return metric: batch bxnxn tensor which is the metric of this fabric term
        """
        
        # Allocate tensors if not already allocated
        if self.metric is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], requires_grad=False,
                                      device=self.device)

        if self.graph_capturable:
            self.metric.zero_().detach_()
            self.force.zero_().detach_()
        else:
            self.metric = torch.zeros_like(self.metric)
            self.force = torch.zeros_like(self.force)

        if not self.is_forcing_policy:
            # Stack position and additional features.
            inputs = torch.cat((x, torch.nn.functional.normalize(xd), features), dim=1)

            # Generate RFFs
            rff_features = self.metric_rff.to_features(inputs.float())
        else:
            # Stack position and additional features.
            inputs = torch.cat((x, features), dim=1)

            # Generate RFFs
            rff_features = self.metric_rff.to_features(inputs.float())

        # Pass inputs through the layers to predict the Cholesky factors for the
        # metric.
        hidden_out = F.elu(self.metric_fc1(rff_features), alpha = self.elu_alpha)
        hidden_out2 = F.elu(self.metric_fc2(hidden_out), alpha = self.elu_alpha)
        output = self.metric_fc3(hidden_out2)

        # Insert factors into lower triangular matrix.
        metric_lower = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=self.device)
        tril_indices = torch.tril_indices(row=x.shape[1], col=x.shape[1], offset=0)
        metric_lower[:, tril_indices[0], tril_indices[1]] = output

        # Ensure the diagonal has strictly positive elements so the resulting metric is P.D.
        metric_diag_elements = torch.diagonal(metric_lower, offset=0, dim1=1, dim2=2)
        metric_diag = torch.diag_embed(metric_diag_elements, offset=0, dim1=1, dim2=2)
        metric_positive_diag =\
            torch.diag_embed(F.elu(metric_diag_elements, alpha = self.elu_alpha) + self.elu_alpha,
                             offset=0, dim1=1, dim2=2)
        metric_lower = metric_lower - metric_diag + metric_positive_diag

        # Reconstruct a positive semi definite metric by multiplying the lower triangular
        # matrix by its transpose and adding a small positive diagonal to ensure
        # positive definiteness.
        # If batch size of metric diag term doesn't match up with the
        # incoming state, then rebuild the diag term.
        if self.metric_small_diag is None or self.metric_small_diag.shape[0] != x.shape[0]:
            small_diag = 1e-2 * torch.ones_like(x)
            self.metric_small_diag = torch.diag_embed(small_diag)

        if self.graph_capturable:
            self.metric.copy_(metric_lower @ torch.transpose(metric_lower, 1, 2) + self.metric_small_diag)
        else:
            self.metric = metric_lower @ torch.transpose(metric_lower, 1, 2) + self.metric_small_diag

        return self.metric

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this fabric term.
        -----------------------------
        @param x: batch bxn tensor of positions
        @param xd: batch bxn tensor of velocities
        @param features: batch bxm tensor of additional features to this fabric term.
        @return force: batch bxn tensor of policy forces
        """

        # If a forcing policy, then create an acceleration-based potential policy.
        if self.is_forcing_policy:
            # Stack the inputs.
            inputs = torch.cat((x, features), dim=1)
            # Create RFFs from inputs.
            rff_features = self.policy_rff.to_features(inputs.float())

            # Function that creates a potential function.
            def potential_function(inputs):
                # Stack position and features (features should be target pose)
                hidden_out = F.elu(self.policy_fc1(inputs), alpha = self.elu_alpha)
                hidden_out2 = F.elu(self.policy_fc2(hidden_out), alpha = self.elu_alpha)
                xdd_not_hd2 = self.policy_fc3(hidden_out2)
                ones = torch.ones((x.shape[1], 1), device=self.device)
                potential = (xdd_not_hd2 @ ones).sum(dim=0)

                return potential
            
            # Ensure the prioritized acceleration can derive from the gradient of a potential function
            # Calculate a scalar function and take its gradient
            potential_force =\
                torch.autograd.functional.jacobian(potential_function,
                    rff_features, create_graph=True).squeeze(0)[:,:self.space_dim]

            # Calculate predicted damping coefficient
            # Stack the inputs.
            inputs_damping = torch.cat((x, xd, features), dim=1)
            
            # Generate RFFs
            rff_features_damping = self.damping_rff.to_features(inputs_damping.float())

            # Pass through the layers.
            x_damp = F.elu(self.damping_fc1(rff_features_damping), alpha = self.elu_alpha)
            x_damp2 = F.elu(self.damping_fc2(x_damp), alpha = self.elu_alpha)
            # NOTE: this damping scalar can never be negative because of the ELU activation
            # followed by the addition of self.elu_alpha. Moreover, additional damping
            # is added because this has shown to be beneficial during the early stages
            # of NGF training.
            # TODO: need to make the 5 here a settable parameter
            damping = F.elu(self.damping_fc3(x_damp2), alpha = self.elu_alpha) +\
                      self.elu_alpha + 5.
            
            # Additionally apply the predicted damping scalar. However, since this
            # damping scalar can be 0, we additionally add a positive damping scalar of 1.
            # which ensures damping is always present, and therefore, the system can converge.
            # NOTE: we use acceleration-based damping here so we multiply through by the metric.
            # NOTE: since the damping scalars are positive and the metric is positive definite,
            # this results in at least positive semi-definite damping matrix at the root.
            # NOTE: Instead of using the metric, one could learn a full positive definite damping
            # matrix.
            self.force.copy_(potential_force + damping * torch.bmm(self.metric, xd.unsqueeze(2)).squeeze())
            #self.force = potential_force + damping * torch.bmm(self.metric, xd.unsqueeze(2)).squeeze()
        else:
            # Stack position, velocity, and features
            inputs = torch.cat((x, torch.nn.functional.normalize(xd), features), dim=1)

            # Generate RFFs
            rff_features = self.policy_rff.to_features(inputs.float())

            # Pass inputs through the layers.
            x_hid = F.elu(self.policy_fc1(rff_features), alpha = self.elu_alpha)
            x1 = F.elu(self.policy_fc2(x_hid), alpha = self.elu_alpha)
            xdd_not_hd2 = self.policy_fc3(x1)

            # Turn into a geometry by multiplying by the velocity squared.
            vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
            xdd = vel_squared * xdd_not_hd2
            self.xdd = xdd

            if self.graph_capturable:
                self.force.copy_(-torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2))
            else:
                self.force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)

