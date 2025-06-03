# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Unit tests for the `fabrics_sim` package version."""

# SRL
import fabrics_sim


def test_fabrics_sim_version() -> None:
    """Test `fabrics_sim` package version is set."""
    assert fabrics_sim.__version__ is not None
    assert fabrics_sim.__version__ != ""
