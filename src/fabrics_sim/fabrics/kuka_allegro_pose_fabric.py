# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion

class KukaAllegroPoseFabric(BaseFabric):
    """
    Creates a fabric for the kuka-allegro that opens up a pose action space for the palm
    and PCA'ed action space for the hand. Includes self-collision, env collision avoidance,
    joint limiting, accel/jerk limiting, speed control, redundancy resolution.
    """
    def __init__(self, batch_size, device, timestep, graph_capturable=True):
        """
        Constructor. Specifies parameter file and constructs the fabric.
        :param batch_size: size of the batch
        :param device: type str that sets the device for the fabric
        """
        # Load parameters
        fabric_params_filename = "kuka_allegro_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        # URDF filpath for allegro
        robot_dir_name = "kuka_allegro"
        robot_name = "kuka_allegro"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        
        self.load_robot(robot_dir_name, robot_name, batch_size)
        
        # Going to set a default config for the cspace attractor that gets
        # used until an actual cspace command comes in
        default_config =\
            torch.tensor([-0.85, -0.50,  0.76,  1.25, -1.76, 0.90, 0.64,
                          0.0, 0.75, 0.75, 0.75,
                          0.0, 0.75, 0.75, 0.75,
                          0.0, 0.75, 0.75, 0.75,
                          1.57, 0.5, 0.5, 0.5], device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        # Store pca matrix for hand
        self._pca_matrix = None

        # Construct the fabric.
        self.construct_fabric()
        
        # Allocate palm pose target tensor (b x (3 + 9))
        # 3 dim for origin target, 12 dim for stacked 3x3 transform target (rx', ry', rx')
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        
        # Storing the target expressed in the taskspace actually used
        self._native_palm_pose_target = None

    def add_joint_limit_repulsion(self):
        """
        Adds forcing joint repulsion to the fabric.
        """

        # Create upper joint limiting
        # Pulling lower joint limits from urdf
        joints = self.urdfpy_robot.joints # this is a list
        upper_joint_limits = []
        for i in range(len(joints)):
            # NOTE: We are only supporting revolute joints right now.
            if joints[i].joint_type == 'revolute':
                upper_joint_limits.append(joints[i].limit.upper)
        # Create upper joint limiting
        # Create taskmap and its container.
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container.
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)
        
        # Create lower joint limiting
        # Pulling lower joint limits from urdf
        lower_joint_limits = []
        for i in range(len(joints)):
            if joints[i].joint_type == 'revolute':
                lower_joint_limits.append(joints[i].limit.lower)

        # Create taskmap and its container.
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container.
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_cspace_attractor(self, is_forcing):
        """
        Add a cspace attractors to the fabric.
        -----------------------------
        :param is_forcing: bool, indicates whether the fabric term will be forcing
                           or not (geometric)
        """
        # Create taskmap and its container.
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create fabric term and add to taskmap container.
        if not is_forcing:
            fabric_name = "cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            # Add it to container list in the root space
            self.add_fabric(taskmap_name, fabric_name, fabric)
        else:
            fabric_name = "forcing_cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['forcing_cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)

        # Add it to container list in the root space
        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_hand_fabric(self):
        # TODO: this will make the PCA space fabric and place an attractor there
        pca_matrix = torch.tensor([[-3.8872e-02,  3.7917e-01,  4.4703e-01,  7.1016e-03,  2.1159e-03,
                                     3.2014e-01,  4.4660e-01,  5.2108e-02,  5.6869e-05,  2.9845e-01,
                                     3.8575e-01,  7.5774e-03, -1.4790e-02,  9.8163e-02,  4.3551e-02,
                                     3.1699e-01],
                                   [-5.1148e-02, -1.3007e-01,  5.7727e-02,  5.7914e-01,  1.0156e-02,
                                    -1.8469e-01,  5.3809e-02,  5.4888e-01,  1.3351e-04, -1.7747e-01,
                                     2.7809e-02,  4.8187e-01,  2.9753e-02,  2.6149e-02,  6.6994e-02,
                                     1.8117e-01],
                                   [-5.7137e-02, -3.4707e-01,  3.3365e-01, -1.8029e-01, -4.3560e-02,
                                    -4.7666e-01,  3.2517e-01, -1.5208e-01, -5.9691e-05, -4.5790e-01,
                                     3.6536e-01, -1.3916e-01,  2.3925e-03,  3.7238e-02, -1.0124e-01,
                                    -1.7442e-02],
                                   [ 2.2795e-02, -3.4090e-02,  3.4366e-02, -2.6531e-02,  2.3471e-02,
                                     4.6123e-02,  9.8059e-02, -1.2619e-03, -1.6452e-04, -1.3741e-02,
                                     1.3813e-01,  2.8677e-02,  2.2661e-01, -5.9911e-01,  7.0257e-01,
                                    -2.4525e-01],
                                   [-4.4911e-02, -4.7156e-01,  9.3124e-02,  2.3135e-01, -2.4607e-03,
                                     9.5564e-02,  1.2470e-01,  3.6613e-02,  1.3821e-04,  4.6072e-01,
                                     9.9315e-02, -8.1080e-02, -4.7617e-01, -2.7734e-01, -2.3989e-01,
                                    -3.1222e-01]], device=self.device)

        self._pca_matrix = torch.clone(pca_matrix.detach())

        # Now stack this PCA matrix with a left block of 0s, which will be used to project against the arm
        # angles. Arm angles are not a dependency here so we wipe them out with 0s.
        pca_matrix = torch.cat([torch.zeros(pca_matrix.shape[0], 7, device=self.device), pca_matrix], dim=1)

        # Create taskmap and its container.
        taskmap_name = "pca_hand"
        taskmap = LinearMap(pca_matrix, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Place an attractor in this space
        fabric_name = "hand_attractor"
        is_forcing = True
        fabric = Attractor(is_forcing, self.fabric_params['hand_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        
        # Add it to container list
        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_palm_points_attractor(self):
        """
        Creates a taskmap of 3 noncollinear points on the gripper and constructs
        a geometric attractor in this space.
        """
        # Set name for taskmap, create it, and add to pool of taskmaps.
        taskmap_name = "palm"
        # TODO: make the control point frames all the points in the gripper head and update code in
        # target point calculation to reflect
        control_point_frames = ["palm_link",
                                "palm_x", "palm_x_neg",
                                "palm_y", "palm_y_neg",
                                "palm_z", "palm_z_neg"]
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, control_point_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
            
        # Create and add geometric attractor
        fabric_name = "palm_attractor"
        is_forcing = True
        fabric = Attractor(is_forcing, self.fabric_params['palm_attractor'],
                           self.device, graph_capturable=self.graph_capturable)

        # Add it to container list
        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_body_repulsion(self):
        """
        Creates body spheres and repulsion between body spheres (self-collision) and also between
        body spheres and environment objects.
        """
        # Create list of frames that will be used to place body spheres at their origins
        collision_sphere_frames = self.fabric_params['body_repulsion']['collision_sphere_frames']

        # List of sphere radii, one for each frame origin
        self.collision_sphere_radii = self.fabric_params['body_repulsion']['collision_sphere_radii']
        
        assert(len(collision_sphere_frames) == len(self.collision_sphere_radii)),\
                "length of link names does not equal length of radii"

        # Declare which body spheres need to avoid collision
        collision_sphere_pairs = self.fabric_params['body_repulsion']['collision_sphere_pairs']
        
        # Calculate the body collision matrix
        collision_matrix = torch.zeros(len(collision_sphere_frames), len(collision_sphere_frames), dtype=int,
                                       device=self.device)

        # If frames for collision sphere pairs were not manually specified, then look for the
        # link prefix pairs so that spheres associated with one link can avoid spheres of the other link
        if len(collision_sphere_pairs) == 0:
            # Find links via prefixes to gather collision spheres for self collision avoidance.
            collision_link_prefix_pairs = self.fabric_params['body_repulsion']['collision_link_prefix_pairs']
            frames_for_prefix1 = None
            frames_for_prefix2 = None
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]

                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])

        for sphere1, sphere2 in collision_sphere_pairs:
            collision_matrix[collision_sphere_frames.index(sphere1), collision_sphere_frames.index(sphere2)] = 1

        # Set name for taskmap, create it, and add to pool of taskmaps.
        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create fabric term and add to taskmap container.
        fabric_name = "repulsion"
        is_forcing = True
        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)
        fabric = BodySphereRepulsion(is_forcing, self.fabric_params['body_repulsion'],
            self.batch_size, sphere_radius, collision_matrix, self.device,
            graph_capturable=self.graph_capturable)

        # Add it to container list
        self.add_fabric(taskmap_name, fabric_name, fabric)

        # Add geometric body repulsion
        fabric_geom = BodySphereRepulsion(False, self.fabric_params['body_repulsion'],
            self.batch_size, sphere_radius, collision_matrix, self.device,
            graph_capturable=self.graph_capturable)
        
        # Add it to container list
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        # Create object that constructs base response and signed distance
        self.base_fabric_repulsion =\
            BaseFabricRepulsion(self.fabric_params['body_repulsion'],
                                self.batch_size,
                                sphere_radius,
                                collision_matrix,
                                self.device)
        
    def add_cspace_energy(self):
        """
        Add a Euclidean cspace energy to the fabric.
        """
        # Add gripper energy.
        taskmap_name = "identity"
        energy_name = "euclidean"
        self.add_energy(taskmap_name, energy_name, EuclideanEnergy(self.batch_size, self._num_joints, self.device))

    def construct_fabric(self):
        """
        Construct the fabric by adding the various geometric, potential, and energy
        components.
        """
        # Add joint limit repulsion
        self.add_joint_limit_repulsion()

        # Add geometric cspace attractor
        self.add_cspace_attractor(False)

        # Add hand attractor
        self.add_hand_fabric()
        
        # Add multi-point gripper attractor
        self.add_palm_points_attractor()

        # Add collision avoidance
        self.add_body_repulsion()

        # Add energy
        self.add_cspace_energy()
    
    def convert_transform_to_points(self):
        """
        Converts gripper pose target to collection of target points in
        gripper frame.
        ------------------------------------------
        :return gripper_targets: bx(3n) Pytorch tensor, where n is number of
                                 gripper points
        """

        palm_transform = torch.zeros(self.batch_size, 4, 4, device=self.device)
        palm_transform[:, 3, 3] = 1.
        palm_transform[:, :3, :3] = torch.transpose(self._palm_pose_target[:, 3:].reshape(self.batch_size, 3, 3), 1, 2)
        palm_transform[:, :3, 3] = self._palm_pose_target[:, :3]

        x_point = torch.zeros(self.batch_size, 4, device=self.device)
        x_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        x_point[:,3] = 1.
        x_neg_point[:,3] = 1.
        x_point[:, 0] = 0.25
        x_neg_point[:, 0] = -0.25

        y_point = torch.zeros(self.batch_size, 4, device=self.device)
        y_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        y_point[:,3] = 1.
        y_neg_point[:,3] = 1.
        y_point[:, 1] = 0.25
        y_neg_point[:, 1] = -0.25

        z_point = torch.zeros(self.batch_size, 4, device=self.device)
        z_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        z_point[:,3] = 1.
        z_neg_point[:,3] = 1.
        z_point[:, 2] = 0.25
        z_neg_point[:, 2] = -0.25

        # Fill in targets
        palm_targets = torch.zeros(self.batch_size, 7 * 3, device=self.device)

        # Origin
        palm_targets[:, :3] = self._palm_pose_target[:, :3]

        # x_axis
        palm_targets[:, 3:6] = torch.bmm(palm_transform, x_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 6:9] = torch.bmm(palm_transform, x_neg_point.unsqueeze(2)).squeeze(2)[:, :3]
        
        # y_axis
        palm_targets[:, 9:12] = torch.bmm(palm_transform, y_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(palm_transform, y_neg_point.unsqueeze(2)).squeeze(2)[:, :3]

        # z_axis
        palm_targets[:, 15:18] = torch.bmm(palm_transform, z_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 18:21] = torch.bmm(palm_transform, z_neg_point.unsqueeze(2)).squeeze(2)[:, :3]

        return palm_targets
    
    def get_sphere_radii(self):
        """
        Returns the radii for the body collision spheres.
        ------------------------------------------
        :return collision_sphere_radii: list of floats containing the radii
        """
        return self.collision_sphere_radii
    
    @property
    def collision_status(self):
        """
        Returns the collision state for each body sphere of the robot
        ------------------------------------------
        :return collision_status: bxn bool Pytorch tensor, b is batch size, n is number of body
                                  spheres
        """
        return self.base_fabric_repulsion.collision_status

    def get_palm_pose(self, cspace_position, orientation_convention):
        """
        Calculates the pose of the palm given joint angles and the given orientation convention.
        ------------------------------------------
        :param cspace_position: bx7 Pytorch tensor, joint position
        :param orientation_convention: str, either "euler_zyx" or "quaternion" (x, y, z, w)
        :return palm_pose: bx6 or bx7 Pytorch tensor that is the pose of the palm, either:
                           (x,y,z,eulerz, eulery, eulerx) or
                           (x,y,z, rx, ry, rz, rw)
        """

        # Calculate the points on the hand given the cspace position
        palm_points, _ = self.get_taskmap("palm")(cspace_position, None)

        # Extract the origin, x-axis, y-axis, and z-axis
        palm_origin = palm_points[:, :3]
        x_point = palm_points[:, 3:6]
        y_point = palm_points[:, 9:12]
        z_point = palm_points[:, 15:18]

        x_axis = torch.nn.functional.normalize(x_point - palm_origin, dim=1)
        y_axis = torch.nn.functional.normalize(y_point - palm_origin, dim=1)
        z_axis = torch.nn.functional.normalize(z_point - palm_origin, dim=1)

        rotation_matrix = torch.zeros(self.batch_size, 3, 3, device=self.device)
        rotation_matrix[:, :, 0] = x_axis
        rotation_matrix[:, :, 1] = y_axis
        rotation_matrix[:, :, 2] = z_axis

       
        orientation = None
        if orientation_convention == "euler_zyx":
            #orientation = transforms.matrix_to_euler_angles(rotation_matrix, "ZYX")
            orientation = matrix_to_euler(rotation_matrix)
        elif orientation_convention == "quaternion":
            #orientation = transforms.matrix_to_quaternion(rotation_matrix)[:, [1, 2, 3, 0]]
            orientation = matrix_to_quaternion(rotation_matrix)[:, [1, 2, 3, 0]]
        else:
            raise ValueError('orientation_convention parameter must be either "euler_zyx" or "quaternion"')

        palm_pose = torch.cat([palm_origin, orientation], dim=-1)

        return palm_pose

    @property
    def pca_matrix(self):
        return self._pca_matrix

    @pca_matrix.setter
    def pca_matrix(self, pca_matrix):
        self._pca_matrix = pca_matrix

    def set_features(self, hand_target, palm_pose_target, orientation_convention,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids,
                     object_indicator,
                     cspace_damping_gain=None):
        """
        Passes the input features to the various fabric terms.
        -----------------------------
        :param hand_target: bx5 Pytorch tensor that sets the desired location in PCA space.
                            Controls the fingers of the Allegro.
        :param palm_pose_target: bxm Pytorch tensor (origin, rotation), where rotation
                            can have 3 elements for Euler "ZYX" angles 
                            (x_angle, y_angle, z_angle) or
                            4 elements for quaternion (x, y, z, w)
        :param orientation_convention: str, either "euler_zyx" or "quaternion" (x, y, z, w)
        :param batched_cspace_position: bx7 Pytorch tensor, current fabric position
        :param batched_cspace_velocity: bx7 Pytorch tensor, current fabric velocity
        :param object_ids: 2D int Warp array referencing object meshes
        :param object_indicator: 2D Warp array of type uint64, indicating the presence
                                 of a Warp mesh in object_ids at corresponding index
                                 0=no mesh, 1=mesh
        """
        self.fabrics_features["pca_hand"]["hand_attractor"] = hand_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config
        
        # Insert translational targets into class tensor for holding the target pose
        self._palm_pose_target[:, :3] = palm_pose_target[:, :3]
        
        # First convert palm target orientation from specified convention to rotation matrix
        if orientation_convention == "euler_zyx":
            assert(palm_pose_target.shape[1] == 6),\
                "Pose target must be of dimensions (batch_size x 6) with Euler convention"
            self._palm_pose_target[:, 3:] =\
                torch.transpose(euler_to_matrix(
                    palm_pose_target[:, 3:]), 1, 2).reshape(self.batch_size, 9)
                #torch.transpose(transforms.euler_angles_to_matrix(
                #    palm_pose_target[:, 3:], "ZYX"), 1, 2).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert(palm_pose_target.shape[1] == 7),\
                "Pose target must be of dimensions (batch_size x 7) with quaternion convention"
            self._palm_pose_target[:, 3:] =\
                torch.transpose(quaternion_to_matrix( # transforms.quaternion_to_matrix(
                    palm_pose_target[:, [6, 3, 4, 5]]), 1, 2).reshape(self.batch_size, 9)
        else:
            raise ValueError('orientation_convention parameter must be either "euler_zyx" or "quaternion"')

        # If multi-point attractor is being used, then convert pose target to targets in the right space
        palm_pose_target = self.convert_transform_to_points()
        
        if self._native_palm_pose_target is None:
            self._native_palm_pose_target = torch.clone(palm_pose_target)
        else:
            self._native_palm_pose_target.copy_(palm_pose_target)
        
        # Pass the gripper target to the gripper attractors and the damping target
        try:
            self.fabrics_features["palm"]["palm_attractor"] =\
                self._native_palm_pose_target
            self.get_fabric_term("palm", "palm_attractor").damping_position =\
                self._native_palm_pose_target
        except:
            raise ValueError('No task map `palm` or `palm_attractor`')
        
        # Calculate current location of body sphere origins and their velocity
        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

        # Calculate signed distance and repulsion response based on body sphere origin
        # position and velocity and objects in the world
        # NOTE: this calculates both self-collision and robot-world collision response
        self.base_fabric_repulsion.calculate_response(body_point_pos,
                                                      body_point_vel,
                                                      object_ids,
                                                      object_indicator)

        # Pass the collision response data into both the forcing and geometric collision
        # avoidance fabric terms.
        self.fabrics_features["body_points"]["repulsion"] =\
            self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] =\
            self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain

