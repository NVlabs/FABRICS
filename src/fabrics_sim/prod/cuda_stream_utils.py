# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Cuda stream utils.

Includes:
- Enabling torch to use warp's custom stream so we don't need to synchronize between them.
- Setting default device for torch.
- Standard setup util that sets them up in a canonical way.

Utils here default to being verbose so it's clear that these global changes are being made.
"""

import torch
import warp as wp


# Global flag determining whether torch is set to use warp streams.
is_cuda_using_warp_streams = False


def setup_torch_to_use_warp_streams(verbose=True):
    """ Set torch to use the custom streams warp uses.

    Setting up torch this way ensures that torch kernels are run on the same (custom) stream warp
    uses. That means torch kernels and warp kernels can be called one after the other and they'll
    automatically pack onto the same stream without requiring synchronization.
    """
    global is_cuda_using_warp_streams
    if is_cuda_using_warp_streams:
        if verbose:
            print("<<< torch is already set to use warp streams >>>")
        return

    if verbose:
        print("<<< setting torch to use warp streams >>>")

    # make Torch use the same (non-default) streams as Warp
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            cuda_stream = wp.get_cuda_device().stream
            torch.cuda.set_stream(torch.cuda.ExternalStream(cuda_stream.cuda_stream))
    is_cuda_using_warp_streams = True


def set_default_torch_device_to_cuda(verbose=True):
    if verbose:
        print("<<< setting torch default device to cuda (floats) >>>")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

