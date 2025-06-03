"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np

class RobotVisualizer():
    def __init__(self, robot_dir_name, robot_name, batch_size, device,
                 robot_body_sphere_radii, robot_body_sphere_position,
                 world_model, vertical_offset, fabric_joint_names,
                 spacing=3.):

        ## Isaac Sim related imports
        from isaacsim import SimulationApp
        self.simulation_app = SimulationApp({"headless": False})

        from omni.isaac.core import SimulationContext
        #from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.world import World
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import XFormPrimView
        
        # Fabrics imports
        from fabrics_sim.utils.path_utils import get_world_path, get_object_urdf_path, get_robot_usd_path

        self.device = device
        self.vertical_offset = vertical_offset
        self.batch_size = batch_size
        self.fabric_joint_names = fabric_joint_names

        self.simulation_context = SimulationContext()

        if World.instance():
            World.instance().clear_instance()
        self.world=World()
        self.world.scene.add_default_ground_plane()

        # Add body spheres
        if robot_body_sphere_radii:
            self.num_spheres = len(robot_body_sphere_radii)
            self.robot_sphere_handles = []
            self.add_robot_spheres(robot_body_sphere_radii)

            self.robot_sphere_view = XFormPrimView(prim_paths_expr="/World/Robot_*/Sphere_*",
                                                   name='robot_sphere_view')
            self.robot_sphere_view.initialize()
            self.world.scene.add(self.robot_sphere_view)

        # add robot articulations
        robot_path = get_robot_usd_path(robot_dir_name, robot_name)

        for i in range(batch_size):
            robot_str = "/World/Robot_" + str(i + 1)
            print('Added robot', str(i+1))
            add_reference_to_stage(usd_path=robot_path, prim_path=robot_str)

        # Now initialize physics
        self.simulation_context.initialize_physics()

        # Create articulation view which we will use to teleport the robot joints
        robot_range = "/World/Robot_[1-9]|[1-9][0-9]{1,3}|9000"
        self.robots_view =\
            ArticulationView(prim_paths_expr=robot_range, name="robots_view")
        self.robots_view.initialize()
        self.robots_view.set_enabled_self_collisions(np.array([False] * batch_size))
        self.robots_view.set_body_disable_gravity(np.zeros((batch_size, self.robots_view.num_bodies)))

        # Add robot view to world
        self.world.scene.add(self.robots_view)
        
        # set root body poses
        #new_positions = np.array([[-1.0, 1.0, 0], [1.0, 1.0, 0]])
        self.robot_base_positions = self.create_grid(spacing = spacing)
        self.robots_view.set_world_poses(positions=self.robot_base_positions)

#        self.robots_view.set_solver_position_iteration_counts(np.full((self.batch_size,), 64))
#        self.robots_view.set_solver_velocity_iteration_counts(np.full((self.batch_size,), 3))

        inertias = np.tile(np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]), (self.batch_size, self.robots_view.num_bodies, 1))
        self.robots_view.set_body_inertias(inertias)

        #print(self.robots_view.get_body_masses())
        #input(self.robots_view.get_body_inertias())

        # Add in fabrics world model objects
        if world_model is not None:
            self.object_handles = []
            self.add_world_objects(world_model)
            self.objects_view = XFormPrimView(prim_paths_expr="/World/Robot_*/Object_*",
                                                   name='objects_view')
            self.objects_view.initialize()
            self.world.scene.add(self.objects_view)

        #print(self.robots_view.dof_names)
        #input(self.robots_view.get_dof_limits())
        
        # take a physics step to create all the handles, etc.
        self.simulation_context.play()
        
        # Since the joint order in isaac sim does not necessarily match the joint order in fabrics,
        # we have to generate a list of joint indices that allow us to rearrange the incoming
        # fabric joint positions such that the joint positions are issued to the correct isaac sim
        # joints
        self.joint_indices =\
            [self.fabric_joint_names.index(joint_name) for joint_name in self.robots_view.dof_names]

    def add_robot_spheres(self, robot_body_sphere_radii):
        from omni.isaac.core.objects import VisualSphere

        for i in range(self.batch_size):
            sphere_handles = []
            for j in range(len(robot_body_sphere_radii)):
                sphere_name = '/World/Robot_' + str(i + 1) + '/Sphere_' + str(j + 1)
                sphere_handles.append(
                    self.world.scene.add(
                        VisualSphere(
                            prim_path=sphere_name,
                            name=sphere_name,
                            position=np.array([0., 0., 0.]),
                            color=np.array([0, 0, 255]),
                            radius=robot_body_sphere_radii[j])
                    )
                )
            self.robot_sphere_handles.append(sphere_handles)

    def add_world_objects(self, world_model):
        from omni.isaac.core.objects import VisualCuboid
        for i in range(self.batch_size):
            # Create objects in world
            # Add visualization box for obstacle box to scene.
            object_names = world_model.get_object_names()
            object_handles = []
            for j in range(len(object_names)):
                object_position = world_model.get_object_transform(object_names[j])
                object_scaling = world_model.get_object_scaling(object_names[j])

                object_name = "/World/Robot_" + str(i + 1) + "/Object_" + str(j + 1)
                position = object_position[:3].detach().cpu().numpy()
                position += self.robot_base_positions[i]
                orientation = np.zeros(4)
                orientation[0] = object_position[6]
                orientation[1] = object_position[3]
                orientation[2] = object_position[4]
                orientation[3] = object_position[5]
                object_handles.append(
                    self.world.scene.add(
                        VisualCuboid(
                            prim_path=object_name,
                            name=object_name,
                            translation=position,
                            orientation=orientation,
                            scale=object_scaling.detach().cpu().numpy(),
                            color=np.array([255, 0, 0]))
                    )
                )
            self.object_handles.append(object_handles)

    def set_robot_sphere_position(self, sphere_position):
        self.robot_sphere_view.set_world_poses(positions=sphere_position)
#        for i in range(self.batch_size):
#            for j in range(len(self.robot_sphere_handles[i])):
#                sphere_pos = sphere_position[i, j]  + self.robot_base_positions[i]
#                self.robot_sphere_handles[i][j].set_world_pose(sphere_pos)

    def create_grid(self, spacing=2.):
        # Set up grid spacing

        # Calculate the grid dimensions (approximate square root)
        num_columns = int(np.ceil(np.sqrt(self.batch_size)))
        num_rows = int(np.ceil(self.batch_size / num_columns))

        # Create a grid of positions
        positions = []
        for i in range(num_rows):
            for j in range(num_columns):
                if len(positions) < self.batch_size:
                    x = (j - num_columns // 2) * spacing
                    y = (i - num_rows // 2) * spacing
                    z = self.vertical_offset
                    positions.append([x, y, z])
                else:
                    break

        # Convert to numpy array
        xyz = np.array(positions)

        return xyz

    def render(self, joint_position, joint_velocity, sphere_position, target_position):
        # set the joint positions for each robot
        self.robots_view.set_joint_positions(joint_position[:, self.joint_indices])
        self.robots_view.set_joint_velocities(joint_velocity[:, self.joint_indices])

        # update the sphere positions
        if sphere_position is not None:
            sphere_world_pos = sphere_position.copy()
            for i in range(self.batch_size):
                sphere_world_pos[i * self.num_spheres : (i + 1) * self.num_spheres, :] +=\
                    np.array([self.robot_base_positions[i, :]])

            self.set_robot_sphere_position(sphere_world_pos)

        # step and render
        self.simulation_context.step(render=True)

        #print('errors', joint_position - self.robots_view.get_joint_positions())
        error = joint_position - self.robots_view.get_joint_positions()

    def close(self):
        self.simulation_context.stop()
        self.simulation_app.close()
