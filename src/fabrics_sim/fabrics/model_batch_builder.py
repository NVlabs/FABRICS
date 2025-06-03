# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import numpy as np

def create_model_batch(builder, batch_size):
    """
    Creates a batched version of Warp's robot model builder without
    copying mesh data over and over again.
    -----------------------------
    @param builder: instantiated ModelBuilder object from Warp
    @param batch_size: size of the batch
    """
    # Rigid bodies
    builder.body_mass *= batch_size
    builder.body_inertia *= batch_size 
    builder.body_com *= batch_size
    builder.body_q *= batch_size
    builder.body_qd *= batch_size

    # Rigid joints
    single_joint_count = builder.joint_count # 35
    single_joint_dof_count = builder.joint_dof_count #9
    #-------------------
    builder.joint_parent *= batch_size
    for i in range(batch_size):
        builder.joint_parent[i * single_joint_count] = builder.joint_parent[0]

        rest_of_batch =\
            np.array(builder.joint_parent[i * single_joint_count + 1: (i+1) * single_joint_count]) +\
            single_joint_count * i

        builder.joint_parent[i * single_joint_count + 1: (i+1) * single_joint_count] = list(rest_of_batch)

    #-------------------
    builder.joint_child *= batch_size
    for i in range(batch_size):
        rest_of_batch =\
            np.array(builder.joint_child[i * single_joint_count: (i+1) * single_joint_count]) +\
            single_joint_count * i

        builder.joint_child[i * single_joint_count: (i+1) * single_joint_count] = list(rest_of_batch)

    #-------------------
    builder.joint_axis *= batch_size
    builder.joint_X_p *= batch_size
    builder.joint_X_c *= batch_size
    builder.joint_q *= batch_size
    builder.joint_qd *= batch_size
    builder.joint_type *= batch_size
    builder.joint_armature *= batch_size
    builder.joint_target_ke *= batch_size
    builder.joint_target_kd *= batch_size
    builder.joint_target *= batch_size
    builder.joint_limit_lower *= batch_size
    builder.joint_limit_upper *= batch_size
    builder.joint_limit_ke *= batch_size
    builder.joint_limit_kd *= batch_size
    
    #-------------------
    builder.joint_q_start *= batch_size
    for i in range(batch_size):
        batch = np.array(builder.joint_q_start[i * single_joint_count : (i+1) * single_joint_count])
        batch += i * len(builder.joint_act)

        builder.joint_q_start[i * single_joint_count : (i+1) * single_joint_count] = list(batch)

    builder.joint_qd_start = builder.joint_q_start.copy()
    builder.articulation_start = list(np.linspace(0, single_joint_count * (batch_size - 1), batch_size, dtype=int)) 
    builder.joint_act *= batch_size
    builder.joint_count *= batch_size
    builder.joint_dof_count *= batch_size
    builder.joint_coord_count *= batch_size
