# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os
import yaml

import torch
import numpy as np
import time

class DisplacementIntegrator():
    """
    Creates the displacement integrator for the fabric
    to produce new positions and velocities.
    """
    def __init__(self, fabric):
        """
        Constructor. Saves off the fabric object.
        """
        self._fabric = fabric
    
    def step(self, joint_position, joint_velocity, joint_accel, timestep):
        """
        Evolves the states of the fabric (position and velocity)
        by following the displacement integration equations.
        -----------------------------
        @param joint_position: batched current position, bxn
        @param joint_velocity: batched current velocity, bxn
        @param timestep: float, amount of time per step
        @return joint_position: batched new position, bxn
        @return joint_velocity: batched new velocity, bxn
        @return joint_accel: batched new acceleration, bxn
        """
        
        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)
        joint_position = joint_position + timestep * joint_velocity + .5 * timestep ** 2 * joint_accel
        joint_velocity = joint_velocity + timestep * joint_accel

        return joint_position, joint_velocity, joint_accel

class ExplicitEulerIntegrator():
    """
    Creates the explicit Euler integrator for the fabric
    to produce new positions and velocities.
    """
    def __init__(self, fabric):
        """
        Constructor. Saves off the fabric object.
        """
        self._fabric = fabric
    
    def step(self, joint_position, joint_velocity, timestep):
        """
        Evolves the states of the fabric (position and velocity)
        by following the displacement integration equations.
        -----------------------------
        @param joint_position: batched current position, bxn
        @param joint_velocity: batched current velocity, bxn
        @param timestep: float, amount of time per step
        @return joint_position: batched new position, bxn
        @return joint_velocity: batched new velocity, bxn
        @return joint_accel: batched new acceleration, bxn
        """
        
        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)
        joint_position = joint_position + timestep * joint_velocity
        joint_velocity = joint_velocity + timestep * joint_accel

        return joint_position, joint_velocity, joint_accel

class SymplecticEulerIntegrator():
    """
    Creates the symplectic Euler integrator for the fabric
    to produce new positions and velocities.
    """
    def __init__(self, fabric):
        """
        Constructor. Saves off the fabric object.
        """
        self._fabric = fabric
    
    def step(self, joint_position, joint_velocity, timestep):
        """
        Evolves the states of the fabric (position and velocity)
        by following the displacement integration equations.
        -----------------------------
        @param joint_position: batched current position, bxn
        @param joint_velocity: batched current velocity, bxn
        @param timestep: float, amount of time per step
        @return joint_position: batched new position, bxn
        @return joint_velocity: batched new velocity, bxn
        @return joint_accel: batched new acceleration, bxn
        """
        
        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)
        joint_velocity = joint_velocity + timestep * joint_accel
        joint_position = joint_position + timestep * joint_velocity

        return joint_position, joint_velocity, joint_accel

class LinearAccelRampIntegrator():
    """
    Integrates following a linear ramp across accelerations
    """
    def __init__(self, fabric):
        """
        Constructor. Saves off the fabric object.
        """
        self._fabric = fabric

    def step(self, joint_position, joint_velocity, joint_accel_prev, timestep):

        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)

        joint_position = (joint_accel - joint_accel_prev) * (timestep ** 2 / 6.) +\
                         joint_accel_prev * (timestep ** 2 / 2.) +\
                         joint_velocity * timestep +\
                         joint_position

        joint_velocity = (joint_accel - joint_accel_prev) * (timestep / 2.) +\
                         joint_accel_prev * timestep +\
                         joint_velocity

        return (joint_position, joint_velocity, joint_accel)
        
class SineAccelRampIntegrator():
    """
    Integrates following a sine ramp across accelerations
    """
    def __init__(self, fabric, dt_max_accel):
        """
        Constructor. Saves off the fabric object.
        """
        self._fabric = fabric
        self._dt_max_accel = dt_max_accel
        self._alpha = np.arcsin(1.) * (1./dt_max_accel)

    def step(self, joint_position, joint_velocity, joint_accel_prev, timestep):

        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)

        # First integrate to dt_max_accel following the sine wave
        joint_position = -(joint_accel - joint_accel_prev) *\
                         (np.sin(self._alpha * self._dt_max_accel) / self._alpha ** 2) + \
                         joint_accel_prev * (self._dt_max_accel ** 2 / 2.) +\
                         joint_velocity * self._dt_max_accel +\
                         joint_position

        joint_velocity = -(joint_accel - joint_accel_prev) *\
                         (np.cos(self._alpha * self._dt_max_accel) / self._alpha) +\
                         joint_accel_prev * self._dt_max_accel +\
                         joint_velocity

        # Now update over the rest of the time interval following a zero-order hold of acceleration
        sub_dt = timestep - self._dt_max_accel
        joint_position = joint_position + sub_dt * joint_velocity + .5 * sub_dt ** 2 * joint_accel
        joint_velocity = joint_velocity + sub_dt * joint_accel

        return (joint_position, joint_velocity, joint_accel)

class FirstOrderAccelIntegrator():
    """
    Integrates by linearizing acceleration at current state and integrating this linear
    acceleration curve over time.
    """
    # NOTE: this should assume that this is only for a batch size of 1, so the actual batch size
    # should then be of 1 + cspace_dim * 2. That is, first in batch is eval at current state and then
    # the remaining cspace * 2 perturbs each element of state (pos, vel) separately and evals at the
    # perturbed state
    def __init__(self, fabric):
        self._fabric = fabric
        self._epsilon = 1e-5

    def step(self, joint_position, joint_velocity, timestep):
        # NOTE: joint_position and joint_velocity should be of batch size 1
        # while the fabric above should have already been set for the below batch size

        # First copy the first state in batch to the rest
        batch_size = 1 + joint_position.shape[1] * 2
        cspace_dim = joint_position.shape[1]
        joint_position = joint_position[0].expand((batch_size, cspace_dim)).contiguous()
        joint_velocity = joint_velocity[0].expand((batch_size, cspace_dim)).contiguous()

        # Perturb states
        cspace_dim = joint_position.shape[1]
        for i in range(cspace_dim):
            # Want to skip the first in batch as we want to evaluate at the current state
            joint_position[i+1, i] = joint_position[i+1, i] + self._epsilon
            joint_velocity[i+1+cspace_dim, i] = joint_velocity[i+1+cspace_dim, i] + self._epsilon

        # Simultaneously eval the acceleration at the current state and perturbed states.
        self._masses, self._forces, self._masses_inv = self._fabric(joint_position, joint_velocity, timestep)
        joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze()
        
        # Calculate acceleration gradient wrt position and velocity
        qdd_grad_q = torch.zeros_like(cspace_dim, cspace_dim)
        qdd_grad_qd = torch.zeros_like(cspace_dim, cspace_dim)
        for i in range(cspace_dim):
            qdd_grad_q[:, i] = (joint_accel[i+1, :] - joint_accel[0,:]) / self._epsilon
            qdd_grad_qd[:, i] = (joint_accel[i+1+cspace_dim, :] - joint_accel[0,:]) / self._epsilon

        # Calculate the vector, A
        A = qdd_grad_q * joint_velocity[0] + qdd_grad_qd * joint_accel[0]

        # Run updates on position, velocity
        joint_position = A * (timestep ** 3 / 6.) +\
                         joint_accel[0] * (timestep ** 2 / 2.) +\
                         joint_velocity[0] * timestep +\
                         joint_position[0]

        joint_velocity = A * (timestep ** 2 / 2.) + joint_accel[0] * timestep + joint_velocity[0]

        return (joint_position, joint_velocity)

