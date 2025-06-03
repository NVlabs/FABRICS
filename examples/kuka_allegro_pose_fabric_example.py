# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Standard Library
import os
import time
import copy
import argparse

# Third party
import torch
import numpy as np

# Fabrics imports
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.visualization.robot_visualizer import RobotVisualizer
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

"""
This example demonstrates how to create and use a kuka-allegro fabric with palm pose and
PCA action spaces. Additional options include:
    1) graph capture of fabric
    2) rendering of the robot and world
    3) rendering of the fabrics collision spheres
    4) setting batch size (number of robots)

Example usage:
python kuka_allegro_pose_fabric_example.py --batch_size=10 --render --cuda_graph
"""

# Reduce print precision
torch.set_printoptions(precision=4)

# Parse arguments
parser = argparse.ArgumentParser(description='Kuka-Allegro fabric example.')
parser.add_argument('--batch_size', type=int, required=True, help='Specify batch size.')
parser.add_argument('--render', action='store_true', help='True to render fabric motion.')
parser.add_argument('--vis_col_spheres', action='store_true', help='True to visualize collision spheres of robot.')
parser.add_argument('--cuda_graph', action='store_true', help='True to enable graph capture of fabric.')
args = parser.parse_args()

# Settings
use_viz = args.render
render_spheres = args.vis_col_spheres
cuda_graph = args.cuda_graph
batch_size = args.batch_size

# Declare device for fabric
device_int = 0
device = 'cuda:' + str(device_int)

# Set the warp cache directory based on device int
warp_cache_dir = ""
initialize_warp(str(device_int))

# This creates a world model that book keeps all the meshes
# in the world, their pose, name, etc.
print('Importing world')
world_filename = 'kuka_allegro_boxes'
max_objects_per_env = 20
world_model = WorldMeshesModel(batch_size=batch_size,
                               max_objects_per_env=max_objects_per_env,
                               device=device,
                               world_filename=world_filename)

# This reports back handles to the meshes which is consumed
# by the fabric for collision avoidance
object_ids, object_indicator = world_model.get_object_ids()

# Control rate and time settings
control_rate = 60.
timestep = 1./control_rate
total_time = 60.

# Create Kuka-Allegro fabric palm pose and finger PCA action spaces
kuka_allegro_fabric =\
    KukaAllegroPoseFabric(batch_size, device, timestep, graph_capturable=cuda_graph)
num_joints = kuka_allegro_fabric.num_joints
            
# Create integrator for the fabric dynamics.
kuka_allegro_integrator = DisplacementIntegrator(kuka_allegro_fabric)

# Create starting states for the robot.
# NOTE: first 7 angles are arm angles, last 16 angles are hand angles
q = torch.tensor([-0.85, -0.50,  0.76,  1.25, -1.76, 0.90, 0.64,
                  0.0,  0.3,  0.3,  0.3,
                  0.0,  0.3,  0.3,  0.3,
                  0.0,  0.3,  0.3,  0.3,
                  0.72383858,  0.60147215,  0.33795027,  0.60845138], device=device)
# Resize according to batch size
q = q.unsqueeze(0).repeat(batch_size, 1).contiguous()
# Start with zero initial velocities and accelerations
qd = torch.zeros(batch_size, num_joints, device=device)
qdd = torch.zeros(batch_size, num_joints, device=device)

# The minimum and maximum values for the PCA targets, and initial targets
hand_mins = torch.tensor([ 0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=device)
hand_maxs = torch.tensor([3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=device)
hand_targets = (hand_maxs - hand_mins) * torch.rand(batch_size, 5, device=device) + hand_mins

# Palm target is (origin, Euler ZYX)
palm_target = np.array([-0.6868,  0.0320,  0.6685, -2.3873, -0.0824,  3.1301])
palm_target = torch.tensor(palm_target, device=device).expand((batch_size, 6)).float()

# Get body sphere raddi
body_sphere_radii = kuka_allegro_fabric.get_sphere_radii()

# Get body sphere locations
sphere_position, _ = kuka_allegro_fabric.get_taskmap("body_points")(q.detach(), None)

# Create visualizer
robot_visualizer = None
if use_viz:
    robot_dir_name = "kuka_allegro_sim"
    robot_name = "kuka_allegro_sim"
    vertical_offset = 0.
    if render_spheres:
        robot_visualizer = RobotVisualizer(robot_dir_name, robot_name, batch_size, device,
                                    body_sphere_radii, sphere_position,
                                    world_model, vertical_offset, kuka_allegro_fabric.get_joint_names())
    else:
        robot_visualizer = RobotVisualizer(robot_dir_name, robot_name, batch_size, device,
                                    None, None,
                                    world_model, vertical_offset, kuka_allegro_fabric.get_joint_names())

# Graph capture
g = None
q_new = None
qd_new = None
qdd_new = None
if cuda_graph:
    # NOTE: elements of inputs must be in the same order as expected in the set_features function
    # of the fabric
    inputs = [hand_targets, palm_target, "euler_zyx",
              q.detach(), qd.detach(), object_ids, object_indicator]
    g, q_new, qd_new, qdd_new =\
        capture_fabric(kuka_allegro_fabric, q, qd, qdd, timestep, kuka_allegro_integrator, inputs, device)

# Loop stepping the fabric forward in time while updating targets, and optionally, rendering
start = time.time()
for i in range(int(control_rate * total_time)):
    # Every two seconds switch targets
    if i % 120 == 0:
        # Update targets for fingers
        hand_targets.copy_((hand_maxs - hand_mins) * torch.rand(batch_size, 5, device=device) + hand_mins)

        # Update targets for palm pose
        palm_target.copy_(torch.rand_like(palm_target))
        palm_target[:, 0] *= -1.
        palm_target[:, 1] -= .5
        palm_target[:, 3:] *= 2. * np.pi

    # Save off current joint states for rendering
    q_prev = q.detach()
    qd_prev = qd.detach()

    # Step the fabric forward in time
    if cuda_graph:
        # Replay through the graph with the above changed inputs
        g.replay()

        # Update the fabric states
        q.copy_(q_new)
        qd.copy_(qd_new)
        qdd.copy_(qdd_new)
    else: 
        # Set the targets
        kuka_allegro_fabric.set_features(hand_targets, palm_target, "euler_zyx",
                                         q.detach(), qd.detach(),
                                         object_ids, object_indicator)
        
        # Integrate fabrics one step producing new position and velocity.
        q, qd, qdd = kuka_allegro_integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)
    
    # Render, albeit at a lower framerate
    if use_viz and (i % 4 == 0):
        # Get body sphere locations reshape into (batch size x num spheres, 3) tensor
        if render_spheres:
            sphere_position = kuka_allegro_fabric.get_taskmap_position("body_points").detach().cpu()
            sphere_position =\
                sphere_position.reshape(batch_size * len(body_sphere_radii), -1).detach().cpu().numpy()
        else:
            sphere_position = None

        robot_visualizer.render(q_prev.detach().cpu().numpy(),
                                qd_prev.detach().cpu().numpy() * 0., # setting to 0 to avoid jitters
                                sphere_position,
                                palm_target.detach().cpu().numpy())
    
    # Get distances to upper, lower joint limits and collision status
    dist_to_upper_limit = kuka_allegro_fabric.get_taskmap_position("upper_joint_limit")
    dist_to_lower_limit = kuka_allegro_fabric.get_taskmap_position("lower_joint_limit")
    collision = kuka_allegro_fabric.collision_status.max().item()

    # Print various signals
    print('time','%.2f' % (i*timestep),
          'wallclock time','%.2f' % (time.time() - start),
          'To upper joint limit','%.3f' % dist_to_upper_limit.min().item(),
          'To lower joint limit', '%.3f' % dist_to_lower_limit.min().item(),
          'Collision', collision,
          'min dist', '%.3f' % kuka_allegro_fabric.base_fabric_repulsion.signed_distance.min())

# Destroy visualizer
if use_viz:
    print('Destroying visualizer')
    robot_visualizer.close()

print('Done')
