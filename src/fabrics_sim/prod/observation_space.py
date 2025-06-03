# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

class ObservationSpace:
    """ Tool for packing individual PyTorch tensors into a single observation space tensor.

    There are two supported use cases:
    1. Where the tensors being packed in are persistent and added just once.
    2. Where the observations are logically the same, but the tensors themselves may differ
       from cycle to cycle.

    Case 1: Persistent observation tensors usage

        observation_space = ObservationSpace(batch_size, device)
        observation_space.add(tensor1)
        observation_space.add(tensor2)
        observation_space.lock()  # Creates the observation tensor.

        while True:
            observations = observation_space.pack()  # Copies the latest data into the observation tensor

    Case 2: Individual observation tensors differ from cycle to cycle

        observation_space = ObservationSpace(batch_size, device)

        while True:
            observation_space.add(tensor1)
            observation_space.add(tensor2)

            # Copies the latest added data into the observation tensor and clears the tensor list.
            # New tensors need to be added each cycle; their semantics should be the same from cycle
            # to cycle. It's a runtime error if new tensors are not added (or the wrong number of
            # dimensions are added).
            observations = observation_space.pack()

    In both cases, at any point observation_space.reset() can be called to reset back to the newly
    constructed state.
    """
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        self.num_dims = 0
        self.obs_tensors = []

        self.observations = None

        self.is_locked = False

    def add(self, obs_tensor):
        if obs_tensor.shape[0] != self.batch_size:
            msg = ("Invalid observation tensor shape:" + str(obs_tensor.shape)
                    + "Expected batch size: %d, received batch size: %d" % obs_tensor.shape[0])
            raise RuntimeError(msg)
        self.obs_tensors.append(obs_tensor)
        self.num_dims += obs_tensor.shape[1]

    def lock(self):
        """ Lock the observation space to the current collection of added tensors.

        Creates the observation tensor. Once locked pack() will just copy into that tensor and
        maintain the list of obseration tensors being copied in.
        """
        self._create_observation_tensor_if_needed()
        self.is_locked = True

    def reset(self):
        self._clear()
        self.observations = None
        self.is_locked = False

    def pack(self):
        if not self.is_locked:
            self._create_observation_tensor_if_needed()

        dim_start_index = 0
        for obs_tensor in self.obs_tensors:
            new_dims = obs_tensor.shape[1]
            self.observations[:, dim_start_index:(dim_start_index+new_dims)] = obs_tensor
            dim_start_index += new_dims
        if dim_start_index != self.observations.shape[1]:
            raise RuntimeError("Packing %d dimensions into an observation tensor with %d observations" % (
                dim_start_index, self.observations.shape[1]))

        if not self.is_locked:
            self._clear()
        return self.observations

    def _create_observation_tensor_if_needed(self):
        if self.observations is None:
            self.observations = torch.zeros(self.batch_size,
                                            self.num_dims,
                                            requires_grad=False,
                                            device=self.device)

    def _clear(self):
        self.num_dims = 0
        self.obs_tensors.clear()



if __name__ == "__main__":
    import argparse
    import sys
    import unittest

    parser = argparse.ArgumentParser("obsevation_space")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Turns on the verbose flag in any given test."),
    )
    args, unrecognized_args = parser.parse_known_args()

    class VerboseTest(unittest.TestCase):
        @classmethod
        def setUpClass(self):
            self.verbose = args.verbose


    class ObservationSpaceTest(VerboseTest):
        @classmethod
        def setUpClass(self):
            super().setUpClass()

        def test_locked(self):
            batch_size = 3
            device = "cuda"

            tensor1 = torch.tensor([.1, .2, .3], device=device).repeat([batch_size, 1])
            tensor2 = torch.tensor([.4, .5], device=device).repeat([batch_size, 1])
            observation_space = ObservationSpace(batch_size, device)
            observation_space.add(tensor1)
            observation_space.add(tensor2)
            observation_space.lock()

            observations = observation_space.pack()
            expected_observations = torch.tensor([.1, .2, .3, .4, .5], device=device).repeat([batch_size, 1])
            if self.verbose:
                print("observations before modification:\n", observations)
                print("expected:\n", expected_observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))

            tensor1[:,1] = torch.tensor([-1., -2., -3.], device=device)
            tensor2[:,0] = torch.tensor([-4., -5., -6.], device=device)

            if self.verbose:
                print("observations after modification before pack:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))  # shouldn't have changed

            # After pack is called, the two updated columns should be visible in the observations.
            observations = observation_space.pack()
            expected_observations[:,1] = tensor1[:,1]
            expected_observations[:,3] = tensor2[:,0]
            if self.verbose:
                print("observations after modification after pack:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))  # now it's changed

            if self.verbose:
                print("resetting")
            observation_space.reset()
            batch_size = 4
            tensor1 = torch.tensor([.1, -.2, .3], device=device).repeat([batch_size, 1])
            tensor2 = torch.tensor([.4, -.5], device=device).repeat([batch_size, 1])
            tensor3 = torch.tensor([.2], device=device).repeat([batch_size, 1])
            observation_space = ObservationSpace(batch_size, device)
            observation_space.add(tensor1)
            observation_space.add(tensor2)
            observation_space.add(tensor3)
            observation_space.lock()
            observations = observation_space.pack()
            expected_observations = torch.tensor([.1, -.2, .3, .4, -.5, .2], device=device).repeat([batch_size, 1])
            if self.verbose:
                print("observations after reset and rebuild:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))

        def test_unlocked(self):
            batch_size = 3
            device = "cuda"

            # Add tensors and pack.
            tensor1 = torch.tensor([.1, .2, .3], device=device).repeat([batch_size, 1])
            tensor2 = torch.tensor([.4, .5], device=device).repeat([batch_size, 1])
            observation_space = ObservationSpace(batch_size, device)
            observation_space.add(tensor1)
            observation_space.add(tensor2)
            observations = observation_space.pack()  # Space is created here.
            expected_observations = torch.tensor([.1, .2, .3, .4, .5], device=device).repeat([batch_size, 1])
            if self.verbose:
                print("observations before modification:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))

            # Re-create the tensors entirely we're packing in.
            tensor1 = torch.tensor([-.1, -.2, -.3], device=device).repeat([batch_size, 1])
            tensor2 = torch.tensor([-.4, -.5], device=device).repeat([batch_size, 1])

            # First try to pack without adding those new tensors in (an error)
            if self.verbose:
                print("trying to pack without adding")
            with self.assertRaises(RuntimeError):
                observations = observation_space.pack()
            if self.verbose:
                print("runtime error was correctly raised")
                print("Now adding the tensors and trying pack again")

            observation_space.add(tensor1)
            observation_space.add(tensor2)
            observations = observation_space.pack()
            expected_observations = -expected_observations
            if self.verbose:
                print("observations after modification:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))

            if self.verbose:
                print("resetting back to original:")
            observation_space.reset()
            tensor1 = torch.tensor([.1, .2, .3], device=device).repeat([batch_size, 1])
            tensor2 = torch.tensor([.4, .5], device=device).repeat([batch_size, 1])
            observation_space = ObservationSpace(batch_size, device)
            observation_space.add(tensor1)
            observation_space.add(tensor2)
            observations = observation_space.pack()  # Space is created here.
            expected_observations = -expected_observations
            if self.verbose:
                print("observations after reset:\n", observations)
            self.assertEqual(0., torch.norm(expected_observations - observations))


    test_args = [sys.argv[0]] + unrecognized_args
    unittest.main(argv=test_args)
