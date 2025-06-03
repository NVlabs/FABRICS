# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os

"""
Implements pathing utility functions.
"""

def get_module_path():
    path = os.path.dirname(__file__)
    return path

def get_root_path():
    path = os.path.dirname(get_module_path())
    return path

def get_params_path():
    root_path = get_root_path()
    path = os.path.join(root_path, "fabric_params")

    return path

def get_root_urdf_path(robot_dir_name):
    root_path = get_root_path()
    path = os.path.join(root_path, "models/robots/urdf/" + robot_dir_name)

    return path

def get_urdf_path(robot_name, urdf_name=None):
    if urdf_name is None:
        urdf_name = robot_name + ".urdf"

    root_path = get_root_path()
    path = os.path.join(root_path, "models/robots/urdf/" + robot_name + "/" + urdf_name)

    return path

def get_robot_urdf_path(robot_dir_name, robot_name):
    root_path = get_root_path()
    path = os.path.join(root_path, "models/robots/urdf/" + robot_dir_name + "/" + robot_name + ".urdf")

    return path

def get_robot_usd_path(robot_dir_name, robot_name):
    root_path = get_root_path()
    path = os.path.join(root_path, "models/robots/USD/" + robot_dir_name + "/" + robot_name + ".usd")

    return path

def get_object_urdf_path(object_name):
    root_path = get_root_path()
    path = os.path.join(root_path, "models/objects/urdf/" + object_name + "/" + object_name + ".urdf")

    return path

def get_world_path(world_name):
    root_path = get_root_path()
    path = os.path.join(root_path, "worlds/" + world_name + ".yaml")

    return path

def get_data_path(data_filename):
    root_path = get_root_path()
    path = os.path.join(root_path, "data/" + data_filename)

    return path

def get_taskmap_model_path(taskmap_model):
    root_path = get_root_path()
    path = os.path.join(root_path, "taskmaps/neural_networks/" + taskmap_model)

    return path
