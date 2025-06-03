# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" URDF import module.

This module is copied from warp.sim.import_urdf. parse_urdf_annotated is based on parse_urdf, but
modified to return additional information important for the kinematics and fabrics calculations
here. Additionally adds an include_collisions flag which can be set to false to prevent constructing
collision data structures which can be slow.
"""

try:
    import urdfpy
except:
    pass

from collections import OrderedDict
import math
import numpy as np
import os
import xml.etree.ElementTree as ET

import warp as wp
from warp.sim.model import Mesh


def urdf_add_collision(builder, link, collisions, density, shape_ke, shape_kd, shape_kf, shape_mu):

    # add geometry
    for collision in collisions:

        origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

        pos = origin[0:3]
        rot = wp.quat_rpy(*origin[3:6])

        geo = collision.geometry

        if geo.box:
            builder.add_shape_box(
                body=link,
                pos=pos,
                rot=rot,
                hx=geo.box.size[0]*0.5,
                hy=geo.box.size[1]*0.5,
                hz=geo.box.size[2]*0.5,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)

        if geo.sphere:
            builder.add_shape_sphere(
                body=link,
                pos=pos,
                rot=rot,
                radius=geo.sphere.radius,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)

        if geo.cylinder:

            # cylinders in URDF are aligned with z-axis, while Warp uses x-axis
            r = wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)

            builder.add_shape_capsule(
                body=link,
                pos=pos,
                rot=wp.mul(rot, r),
                radius=geo.cylinder.radius,
                half_width=geo.cylinder.length*0.5,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)

        if geo.mesh:

            for m in geo.mesh.meshes:
                faces = []
                vertices = []

                for v in m.vertices:
                    vertices.append(np.array(v))

                for f in m.faces:
                    faces.append(int(f[0]))
                    faces.append(int(f[1]))
                    faces.append(int(f[2]))

                mesh = Mesh(vertices, faces)

                builder.add_shape_mesh(
                    body=link,
                    pos=pos,
                    rot=rot,
                    mesh=mesh,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu)


class UrdfInfo:
    def __init__(self, cspace_names, link_index_map, cspace2link, link2cspace, cspace_joint_limits):
        self.cspace_names = cspace_names
        self.link_index_map = link_index_map
        self.cspace2link = cspace2link
        self.link2cspace = link2cspace
        self.cspace_joint_limits = cspace_joint_limits

        self.cspace_name2index_map = {}
        for i, name in enumerate(self.cspace_names):
            self.cspace_name2index_map[name] = i


def expand_link(branches, link_name, joint_list):
    """ Expands the named link by looking up the set of joints in the branches. Adds them recursively
    to the joint_list in depth first order.
    """
    if link_name not in branches:
        # Base case.
        return
    else:
        edges = branches[link_name]
        for joint in edges:
            joint_list.append(joint)
            expand_link(branches, joint.child, joint_list)


class KinematicTree(object):
    """ Simple representation of the kinematic tree stored in a urdfpy URDF object.

    Represents the tree simply as a map from link name to the list of branches extending from that link.
    The branches for each link are listed in the order they're specified in the URDF.
    """
    def __init__(self, robot):
        self.robot = robot
        self.branches = {}
        for joint in robot.joints:
            self.get_link_branches(joint.parent).append(joint)

    def get_link_branches(self, name):
        """ Get the list of branches for the named link. If that link isn't currently in the dict,
        populates the entry as an empty list and returns that.
        """
        if name not in self.branches:
            b = []
            self.branches[name] = b
            return b
        else:
            return self.branches[name]

    def get_depth_first_joints(self):
        """ Returns the list of joints in depth first order.
        """
        joint_list = []
        expand_link(self.branches, self.robot.base_link.name, joint_list)
        return joint_list


#def get_link_to_cspace_map(joint_list):
#    """ Just step through the joints and increment whenever the joint isn't fixed.
#    That'll give a map from link to cspace which we can use for lookup.
#    """
#    link_to_cspace_map = []
#    cspace_index = 0
#    for joint in joint_list:
#        if joint.
#    return link_to_cspace_map


def parse_urdf_annotated(
        filename,
        builder,
        xform,
        floating=False,
        density=0.0,
        stiffness=100.0,
        damping=10.0,
        armature=0.0,
        shape_ke=1.e+4,
        shape_kd=1.e+3,
        shape_kf=1.e+2,
        shape_mu=0.25,
        limit_ke=100.0,
        limit_kd=10.0,
        include_collisions=True,
        verbose=False):

    if verbose:
        print("urdfpy loading:", filename)
    robot = urdfpy.URDF.load(filename)
    kinematic_tree = KinematicTree(robot)
    depth_first_joints = kinematic_tree.get_depth_first_joints()
    if verbose:
        for i, joint in enumerate(depth_first_joints):
            print("{}) <{}> -> <{}>".format(i, joint.parent, joint.child))

    cspace_names = []   # collects the cspace names
    link_index = OrderedDict() # maps from link name -> link index. Also, stores names in link index order.
    cspace2link = []    # for each cspace dim, contains the corresponding link index
    link2cspace = []
    cspace_joint_limits = []

    if verbose:
        print("add articulation")
    builder.add_articulation()

    # import inertial properties from URDF if density is zero
    if verbose:
        print("importing inertial props")
    if density == 0.0:
        com = urdfpy.matrix_to_xyz_rpy(robot.base_link.inertial.origin)[0:3]
        I_m = wp.mat33(robot.base_link.inertial.inertia)
        m = robot.base_link.inertial.mass
    else:
        com = np.zeros(3)
        I_m = wp.mat33(np.zeros((3, 3)))
        m = 0.0

    # add base
    if floating:
        if verbose:
            print("is floating")
        root = builder.add_body(origin=wp.transform_identity(),
                                armature=0.,
                                joint_armature=armature,
                                com=com,
                                I_m=I_m,
                                m=m)
        builder.add_joint_free(child=root)

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]
        urdf_add_collision(
            builder, root, robot.links[0].collisions, density, shape_ke, shape_kd, shape_kf, shape_mu)
    else:
        if verbose:
            print("not floating")
            print("  <adding body>")

        root = builder.add_body(origin=wp.transform_identity())
        
        builder.add_joint_fixed(parent=-1,
                                child=root,
                                parent_xform=xform)

        if include_collisions:
            if verbose:
                print("  <adding collision>")
            urdf_add_collision(
                builder, root, robot.links[0].collisions, 0.0, shape_ke, shape_kd, shape_kf, shape_mu)
            if verbose:
                print("  <done>")

    link_index[robot.base_link.name] = root
    link2cspace.append(-1) # No corresponding cspace dim for the root link.

    # add children
    if verbose:
        print("adding children")
    for joint in depth_first_joints:
        if verbose:
            print("joint:", joint.name)

        type = None
        axis = (0.0, 0.0, 0.0)

        if joint.joint_type == "revolute" or joint.joint_type == "continuous":
            type = wp.sim.JOINT_REVOLUTE
            axis = joint.axis
        if joint.joint_type == "prismatic":
            type = wp.sim.JOINT_PRISMATIC
            axis = joint.axis
        if joint.joint_type == "fixed":
            type = wp.sim.JOINT_FIXED
        if joint.joint_type == "floating":
            type = wp.sim.JOINT_FREE

        if joint.parent not in link_index:
            # All parents encountered should be already available in the link_index structure because
            # depth_first_joints is in depth-first order.
            raise RuntimeError("Parent not found: {}".format(joint.parent))
        parent = link_index[joint.parent]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)
        pos = origin[0:3]
        rot = wp.quat_rpy(*origin[3:6])

        lower = -1.e+3
        upper = 1.e+3
        damping = 0.0

        # limits
        if joint.limit:
            if joint.limit.lower != None:
                lower = joint.limit.lower
            if joint.limit.upper != None:
                upper = joint.limit.upper

        # damping
        if joint.dynamics:
            if joint.dynamics.damping:
                damping = joint.dynamics.damping

        if density == 0.0:
            com = urdfpy.matrix_to_xyz_rpy(robot.link_map[joint.child].inertial.origin)[0:3]
            I_m = wp.mat33(robot.link_map[joint.child].inertial.inertia)
            m = robot.link_map[joint.child].inertial.mass
        else:
            com = np.zeros(3)
            I_m = wp.mat33(np.zeros((3, 3)))
            m = 0.0

        # add link
        if verbose:
            print("  <adding body>")

        start_joint_coord_count = builder.joint_coord_count
        link = builder.add_body(
            origin=wp.transform_identity(),
            armature=0.,
            com=com,
            I_m=I_m,
            m=m)

        if type==wp.sim.JOINT_FIXED:
            builder.add_joint_fixed(parent=parent,
                                    child=link,
                                    parent_xform=wp.transform(pos, rot))
        elif type==wp.sim.JOINT_REVOLUTE:
            builder.add_joint_revolute(parent=parent,
                                       child=link,
                                       parent_xform=wp.transform(pos, rot),
                                       child_xform=wp.transform(),
                                       axis=axis,
                                       limit_lower=lower,
                                       limit_upper=upper,
                                       limit_ke=limit_ke,
                                       limit_kd=limit_kd)
        else:
            raise RuntimeError("Joint type not currently supported.")

        end_joint_coord_count = builder.joint_coord_count
        if end_joint_coord_count != start_joint_coord_count:
            # We'll enter here for each new joint. This is an added cspace dimension.
            cspace_names.append(joint.name)
            cspace2link.append(builder.joint_count-1)  # Minus 1 to get index.
            cspace_joint_limits.append((lower, upper))

        link2cspace.append(len(cspace_names)-1)

        if include_collisions:
            # add collisions
            if verbose:
                print("  <adding collision>")
            urdf_add_collision(
                builder, link, robot.link_map[joint.child].collisions, density, shape_ke, shape_kd, shape_kf, shape_mu)
            if verbose:
                print("  <done>")

        # add ourselves to the index
        link_index[joint.child] = link

    if verbose:
        print("done loading urdf")

    return UrdfInfo(cspace_names, link_index, cspace2link, link2cspace, cspace_joint_limits)
