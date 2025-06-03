# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a taskmap container that holds the task map, and various fabric and energy terms.
"""

import torch
import time

class TaskmapContainer():
    def __init__(self, taskmap_name, taskmap, graph_capturable):
        self._taskmap_name = taskmap_name
        self._taskmap = taskmap

        self.graph_capturable = graph_capturable

        # Additional features associated with taskmap
        self._features = None

        # Dictionaries of fabrics and energies
        self._fabrics = dict()
        self._energies = dict()
        #self.stream = torch.cuda.Stream(device='cuda')

        # Hold onto the taskspace evals
        self._x = None
        self._jac = None

        # Pre-delare some tensors
        self.M_leaf = None
        self.potential_force_lhs_leaf = None
        self.geometric_force_lhs_leaf = None

        self.M_energy_leaf = None
        self.energy_force_lfs_leaf = None
        self.energy_scalars = None

    # TODO: need to figure out how to use streams properly once supported by Warp.
    #def get_stream(self):
    #    return self.stream

    @property
    def name(self):
        return self._taskmap_name

    @property
    def x(self):
        return self._x

    @property
    def jac(self):
        return self._jac

    @property
    def taskmap(self):
        return self._taskmap

    def set_features(self, features):
        self._features = features

    def get_fabric(self, fabric_name):
        return self._fabrics[fabric_name]

    def add_fabric(self, fabric_name, fabric):
        self._fabrics[fabric_name] =  fabric

    def add_energy(self, energy_name, energy):
        self._energies[energy_name] = energy

    def eval_taskmap(self, q):
        self._x, self._jac = self._taskmap(q, self._features)
        return (self._x, self._jac)

    def eval_fabrics(self, x, xd, fabric_features_dict, external_force):
        # List of metrics and forces associated with this leaf space.
        #M_leaf =  []
        #potential_force_lhs_leaf =  []
        #geometric_force_lhs_leaf =  []

        # Allocate memory
        if self.M_leaf is None:
            batch_size = x.shape[0]
            num_dim = x.shape[1]
            self.M_leaf = torch.zeros(batch_size, num_dim, num_dim, device=x.device)
            self.potential_force_lhs_leaf = torch.zeros(batch_size, num_dim, device=x.device)
            self.geometric_force_lhs_leaf = torch.zeros(batch_size, num_dim, device=x.device)

        # Zero out tensors
        if self.graph_capturable:
            self.M_leaf.zero_().detach_()
            self.potential_force_lhs_leaf.zero_().detach_()
            self.geometric_force_lhs_leaf.zero_().detach_()
        else:
            self.M_leaf = torch.zeros_like(self.M_leaf)
            self.potential_force_lhs_leaf =\
                torch.zeros_like(self.potential_force_lhs_leaf)
            self.geometric_force_lhs_leaf =\
                torch.zeros_like(self.geometric_force_lhs_leaf)

        # Cycle through fabrics in this task space, generating their responses.
        for fabric_name, fabric in self._fabrics.items():
            # Evaluate fabric with associated features.
            M_term, force_term = fabric(x, xd, fabric_features_dict[fabric_name])
            #M_leaf.append(M_term)

            if self.graph_capturable:
                self.M_leaf.add_(M_term)
            else:
                self.M_leaf = self.M_leaf + M_term

            if fabric.is_forcing_policy:
                #potential_force_lhs_leaf.append(force_term)
                if self.graph_capturable:
                    self.potential_force_lhs_leaf.add_(force_term)
                else:
                    self.potential_force_lhs_leaf =\
                        self.potential_force_lhs_leaf + force_term
            else:
                #geometric_force_lhs_leaf.append(force_term)
                if self.graph_capturable:
                    self.geometric_force_lhs_leaf.add_(force_term)
                else:
                    self.geometric_force_lhs_leaf =\
                        self.geometric_force_lhs_leaf + force_term
        
        # Sum up the metrics
        #M = None
        #if len(M_leaf) > 0:
        #    M = torch.sum(torch.stack(M_leaf, 3), 3)

        # Sum up potential force
        potential_force_lhs = None
        geometric_force_lhs = None
        
        # Add external force to potential force
        if external_force is not None:
            #potential_force_lhs_leaf.append(external_force)
            if self.graph_capturable:
                self.potential_force_lhs_leaf.add_(external_force)
            else:
                self.potential_force_lhs_leaf =\
                    self.potential_force_lhs_leaf + external_force

        #if len(potential_force_lhs_leaf) > 0:
        #    potential_force_lhs = torch.sum(torch.stack(potential_force_lhs_leaf, 2), 2)

        # Sum up geometric force
        #if len(geometric_force_lhs_leaf) > 0:
        #    geometric_force_lhs = torch.sum(torch.stack(geometric_force_lhs_leaf, 2), 2)

        #return (M, potential_force_lhs, geometric_force_lhs)
        return (self.M_leaf, self.potential_force_lhs_leaf, self.geometric_force_lhs_leaf)

    def eval_energies(self, x, xd):
        #M_leaf = []
        #force_lhs_leaf = []
        #energy_scalars = []

        if self.M_energy_leaf is None:
            batch_size = x.shape[0]
            num_dim = x.shape[1]
            self.M_energy_leaf = torch.zeros(batch_size, num_dim, num_dim, device=x.device)
            self.energy_force_lfs_leaf = torch.zeros(batch_size, num_dim, device=x.device)
            self.energy_scalars = torch.zeros(batch_size, 1, device=x.device)

        # Zero out tensors
        if self.graph_capturable:
            self.M_energy_leaf.zero_().detach_()
            self.energy_force_lfs_leaf.zero_().detach_()
            self.energy_scalars.zero_().detach_()
        else:
            self.M_energy_leaf = torch.zeros_like(self.M_energy_leaf)
            self.energy_force_lfs_leaf = torch.zeros_like(self.energy_force_lfs_leaf)
            self.energy_scalars = torch.zeros_like(self.energy_scalars)

        # Cycle through fabrics in this task space, generating their responses.
        for energy_name, energy in self._energies.items():
            M_term, force_term, energy_scalars_term = energy(x, xd)
            #M_leaf.append(M_term)
            #force_lhs_leaf.append(force_term)
            #energy_scalars.append(energy_scalars_term)
            if self.graph_capturable:
                self.M_energy_leaf.add_(M_term)
                self.energy_force_lfs_leaf.add_(force_term)
                self.energy_scalars.add_(energy_scalars_term)
            else:
                self.M_energy_leaf = self.M_energy_leaf + M_term
                self.energy_force_lfs_leaf = self.energy_force_lfs_leaf + force_term
                self.energy_scalars = self.energy_scalars + energy_scalars_term
        
        # Sum up the metrics
#        M = None
#        force_lhs = None
#        energy = None
#        if len(M_leaf) > 0:
#            M = torch.sum(torch.stack(M_leaf, 3), 3)
#            force_lhs = torch.sum(torch.stack(force_lhs_leaf, 2), 2)
#            energy = torch.sum(torch.stack(energy_scalars, 1), 1)

        #return (M, force_lhs, energy)
        return (self.M_energy_leaf, self.energy_force_lfs_leaf, self.energy_scalars)
