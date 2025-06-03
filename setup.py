# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Fabrics Sim package setuptools."""

#
# References:
# * https://setuptools.pypa.io/en/latest/setuptools.html#setup-cfg-only-projects

import os

# Third Party
import setuptools

INSTALL_REQUIRES = []

# NOTE: check is this is an ARM device. If so, then don't install
# pytorch because currently you can't pip install pytorch with
# cuda support. Assumes you already have cuda support pytorch
# installed.
if os.uname().machine == 'aarch64':
    INSTALL_REQUIRES = [
        "warp-lang>=1.5.0",
        "pyyaml",
        "urdfpy",
        "numpy>=1.23.5"
        ]
# Install dependencies including pytorch
else:
    INSTALL_REQUIRES = [
        "torch>=2.4.0",
        #"torchvision>=0.15.2",
        #"torchaudio==2.0.2", # NOTE: don't think we need this for now.
        "warp-lang>=1.5.0",
        "pyyaml",
        "urdfpy",
        "numpy>=1.23.5"
        ]

setuptools.setup(install_requires=INSTALL_REQUIRES)
