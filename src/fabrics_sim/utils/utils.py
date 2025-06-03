# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements several utility functions.
"""

import torch
import warp as wp

def initialize_warp(warp_cache_name):
    """
    Explicitly setting the directory for codegen and compilation. Need this for multi-gpu settings.
    See https://omniverse.gitlab-master-pages.nvidia.com/warp/basics.html#example-cache-management
    ----------------------------------------------
    :param warp_cache_name: str, subdirectory name for placing generated code and compilation.
    """
    import os
    import warp as wp

    wp.config.kernel_cache_dir = '/tmp/.cache/warp/' + wp.config.version + "/warpcache_" + warp_cache_name
    wp.init()

    # clear kernel cache (forces fresh kernel builds every time)
    #wp.build.clear_kernel_cache()

def capture_fabric(fabric, q, qd, qdd, timestep, fabric_integrator, inputs, device):
    # Set stream
    torch_stream = torch.cuda.Stream(device=device)

    # make warp use the same stream
    warp_stream = wp.stream_from_torch(torch_stream)

    # Warmup
    with torch.cuda.stream(torch_stream) and wp.ScopedStream(warp_stream):
        for i in range(3):
            fabric.set_features(*inputs)
            q_new, qd_new, qdd_new = fabric_integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)

    # capture
    g = torch.cuda.CUDAGraph()

    with wp.ScopedStream(warp_stream), torch.cuda.graph(g, stream=torch_stream):
        # Set features/actions
        fabric.set_features(*inputs)
        
        # Integrate the fabric forward one step
        q_new, qd_new, qdd_new = fabric_integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)

    return (g, q_new, qd_new, qdd_new)

# TODO: need to test this. v should be a function of u
def jvp(v, u, w, retain_graph=False, create_graph=False):
    """
    Calculates batched Jacobian-vector product, \partial_u v * w.
    -----------------------------
    @param v: batched vector with shape (b, n), b is batch size
    @param u: batched vector with shape (b, m)
    @param w: batched vector with shape (b, m)
    @return jvp_result: batched vector with shape (b, n)
    """
    
    assert torch.is_tensor(v) and torch.is_tensor(u) and torch.is_tensor(w)
    u.grad = torch.zeros(u.shape, requires_grad=True, device='cuda') 
    v.backward(torch.ones(v.shape, device='cuda'), retain_graph=retain_graph, create_graph=create_graph)
    jvp_result = torch.clone(u.grad)

    return jvp_result

def jacobian(v, u, retain_graph=True, create_graph=True):
    """
    Calculates batched Jacobian, \partial_u v
    -----------------------------
    @param v: batched vector with shape (b, n), b is batch size
    @param u: batched vector with shape (b, m)
    @return jacobian: batched tensor with shape (b, n, m)
    """

    # Calculate another jacobian which becomes the root mass
    jacobian = torch.zeros(v.shape[0], v.shape[1], u.shape[1], device='cuda')
    g_vec = torch.zeros(v.shape, device='cuda')

    #jacobian_per_leaf_coordinate = []

    if u.grad is not None:
        with torch.no_grad():
            u.grad *= 0.
    u.grad = torch.zeros(u.shape, requires_grad=True, device='cuda') 
    for i in range(v.shape[1]):
        g_vec *= 0.
        g_vec[:, i] = 1.0
        v.backward(g_vec, retain_graph=retain_graph, create_graph=create_graph)
        jacobian[:,i,:] = torch.clone(u.grad).detach()
        #jacobian_per_leaf_coordinate.append(torch.clone(u.grad))
        with torch.no_grad():
            u.grad *= 0.

    #jacobian = torch.stack(jacobian_per_leaf_coordinate, 1)

    return jacobian
            
#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
#start.record()
## do stuff
#end.record()
#torch.cuda.synchronize()
#print('timey map', start.elapsed_time(end), 'name', container.name)

def check_numerical_jacobian(func, x_coord_index, q):
    #x,_ = func(q)
    x = func(x_coord_index, q)

    # Check Jacobian
    eps = 1e-4
    jac = torch.zeros((q.shape[0], x.shape[1], q.shape[1]), device='cuda')

#    for i in range(q.shape[0]):
#        for j in range(q.shape[1]):
#            q_copy = torch.clone(q).detach()
#            q_copy[i,j] += eps
#            x_new = func(q_copy)
#            jac[i,j] = (x_new[0] - x[0]) / eps
#            q_copy[i,j] -= eps
   
    for i in range(q.shape[0]):
        for j in range(x.shape[1]):
            for k in range(q.shape[1]):
                q_copy = torch.clone(q).detach()
                q_copy[i,k] += eps
                #x_new, _ = func(q_copy)
                x_new = func(x_coord_index, q_copy)
                jac[i,j,k] = (x_new[i,j] - x[i,j]) / eps
                q_copy[i,k] -= eps

    return jac

#
## TODO:
#def check_numerical_velocity():
#            eps_check = 1e-4
#            x_eps = taskmap(q + eps_check * qd)
#
#            xd_check = (x_eps - x) / eps_check
#
#            print('xd', xd)
#            print('xd check', xd_check)
#            input('paused')
#
## TODO:
#def check_numerical_curvature_force():
