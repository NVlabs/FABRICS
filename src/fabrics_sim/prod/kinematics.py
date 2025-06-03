# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Kinematics tooling including efficient kernels.
"""

import torch
import numpy as np

import warp as wp
from fabrics_sim.prod.import_urdf import parse_urdf_annotated as parse_urdf
from fabrics_sim.prod.cuda_stream_utils import setup_torch_to_use_warp_streams


def extract_link_index_path(link_index, joint_parents):
    """ Extracts the specified link's path from the root to the given link.

    Note: Each link has a joint connecting it to its parent link. The joint_parents array is
    therefore indexed by the link indices.
    """
    path = [link_index]

    # Collect up the link indices in the path from the link_index to the root.
    while True:
        if link_index == 0:
            # We're at the root.
            break

        parent_index = joint_parents[link_index]
        path.append(parent_index)

        link_index = parent_index

    # Then reverse the path so it's from the root to the link index.
    path.reverse()
    return path


def extract_link_paths(joint_parents):
    """ Extracts the link index paths to a given link for each link.

    joint_parents is an array of parent link indices for each link (indices correspond to link
    indices). Runs an O(n^2) algorithm for retrieving the paths. These kinematic trees are small, so
    while algorithmically inefficient, it's still very fast.
    """
    paths = []
    num_links = len(joint_parents)
    for link_index in range(num_links):
        path = extract_link_index_path(link_index, joint_parents)
        paths.append(path)

    return paths


def make_ancestory_matrix_from_paths(link_paths_numpy, device):
    """ Create a link ancestory matrix

    The matrix contains 0 and 1. Let (i,j) be the (row,col) indices. An entry of 1 means link i is
    an ancestor of link j.
    """
    num_links = len(link_paths_numpy)
    link_ancestory_matrix = wp.zeros(shape=(num_links, num_links), dtype=int, device=device)
    link_ancestory_matrix_torch = wp.torch.to_torch(link_ancestory_matrix)

    # We'll look at the link paths to a given link. If a link i appears along the link path of a
    # given link j, then i s an ancessor of j as long as i != j. That last condition only holds for
    # the last index of the path, so we just skip it.
    for j in range(num_links):
        link_path = link_paths_numpy[j]
        for i in link_path[:-1]:
            link_ancestory_matrix_torch[i,j] = 1 

    return link_ancestory_matrix


def paths_to_warp(paths, device):
    """ Convert the list of paths into a warp array.

    The warp array with be num_links x max_path_length+1. The ith link's path will be packed into
    the left-most portion of the ith row. The first element of the row is the length of the paths,
    followed by the link indices along the path.
    """
    num_links = len(paths)
    max_len = max([len(p) for p in paths])
    torch_paths = torch.zeros(num_links, max_len+1, dtype=torch.int32, device=device)
    for link_index in range(num_links):
        path = paths[link_index]
        torch_paths[link_index, 0] = len(path)
        for path_index in range(len(path)):
            torch_paths[link_index, path_index+1] = path[path_index]

    return wp.torch.from_torch(torch_paths)


""" Efficient kernels for computing the forward kinematics of the robot.

The kinematics data for the robot is stored only once and broadcast to each thread. See discussion here:

    https://www.quora.com/In-CUDA-can-you-read-a-memory-address-in-global-memory-at-the-same-time-for-different-threads

In CPU code it's dangerous to have separate threads read from the same memory location, so
everything needs to be copied once per thread. But on GPUs it's not only possible to read from the
same memory location, it's also faster. The hardware pulls the data only once from main memory, then
both has it in cache and very quickly broadcasts it to each thread in a given warp. 

Interpreting the kinematics data:

Each joint is a connection between a parent link and a child link. The parent link, joint, and child
link all have their own local coordinate system. (The joint frame is effective equivalent to the
child frame, but at zero rotation. The child frame accounts for the rotation.)

Transform naming shorthand X_<b><a> denotes a transform from <a> coords to <b> coords.  Think of it
as going from right to left z = X_<b><a> y transforms y from coordinates <a> to z in coordinates
<b>. In newer kernels, we use a more explicit notation X_<a>2<b> which should be read semantically as
transforming from <a> coordinates to <b> coordinates.

In this case, we have labels:
- w: world coords
- j: joint coords
- p: parent link coords
- c: child link coords

(Again, newer kernels will more explicitly write out the names, such as X_child2world (which would
be the child link's frame in world coordinates).)

For example, X_pj will transform from joint coordintes to parent coordinates. The final transform we
want is X_wc, the child link in world coordinates. We calculate it using

  X_wc = X_wp * X_pj * X_jc

Read from right to left, these transforms incrementally transform from child coordinates to joint
coordinates, then from joint coordinates to parent coordinates, then finally from parent coordinates
to world coordinates.

X_jc is defined by the rotation around the joint and is calculated here. X_pj is given from the
data. And X_wp is the previously calculated forward kinematics transform of the parent link.

Question: Don't we also need a transform from from the child coordinates to where the joint lines up
with the link? DH makes this easy, but that also defines the axis, which we're assuming is passed in
here.

Answer: https://urdfpy.readthedocs.io/en/latest/generated/urdfpy.Joint.html#urdfpy.Joint
"joint.origin – The pose of the child link with respect to the parent link’s frame. The joint frame
is defined to be coincident with the child link’s frame, so this is also the pose of the joint frame
with respect to the parent link’s frame."

This is at zero joint angle. Then the joint value makes this shift.
"""

@wp.kernel
def link_transforms_kernel(
        # inputs
        cspace_q: wp.array(dtype=float, ndim=2),
        num_links: wp.int32,
        joint_types: wp.array(dtype=int, ndim=1),
        joint_parents: wp.array(dtype=int, ndim=1),
        joint_transforms_in_parent_coords: wp.array(dtype=wp.transform, ndim=1),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        link2cspace: wp.array(dtype=int, ndim=1),
        # outputs
        link_transforms: wp.array(dtype=wp.transform, ndim=2)):
    """ Calculates the forward kinematic link transforms for each link. This kernel processes the
    entire robot in a single thread, looping through the links.
    """
    tid = wp.tid()

    for link_index in range(num_links):
        parent_index = joint_parents[link_index]
        X_wp = wp.transform_identity()
        if parent_index >= 0:
            X_wp = link_transforms[tid, parent_index]

        joint_axis = local_joint_axes[link_index]
        X_pj = joint_transforms_in_parent_coords[link_index]
        
        joint_type = joint_types[link_index]
        if joint_type == wp.sim.JOINT_REVOLUTE:
            cspace_index = link2cspace[link_index]
            q = cspace_q[tid, cspace_index]
            X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(joint_axis, q))
        elif joint_type == wp.sim.JOINT_FIXED:
            X_jc = wp.transform_identity()

        X_wc = X_wp * X_pj * X_jc
        link_transforms[tid, link_index] = X_wc


@wp.kernel
def link_transforms_multithreaded_kernel(
        # inputs
        cspace_q: wp.array(dtype=float, ndim=2),
        joint_types: wp.array(dtype=int, ndim=1),
        joint_parents: wp.array(dtype=int, ndim=1),
        joint_transforms_in_parent_coords: wp.array(dtype=wp.transform, ndim=1),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        link_paths: wp.array(dtype=int, ndim=2),
        link2cspace: wp.array(dtype=int, ndim=1),
        # outputs
        link_transforms: wp.array(dtype=wp.transform, ndim=2)):
    """ This version of the kernel computes each link transform in a separate thread rather than all
    of them in a sequential loop.
    """
    batch_index, target_link_index = wp.tid()

    X_wc = wp.transform_identity()

    path_length = link_paths[target_link_index, 0]
    for i in range(path_length):
        link_index = link_paths[target_link_index, i+1]
        joint_axis = local_joint_axes[link_index]
        X_pj = joint_transforms_in_parent_coords[link_index]
        
        joint_type = joint_types[link_index]
        if joint_type == wp.sim.JOINT_REVOLUTE:
            cspace_index = link2cspace[link_index]
            q = cspace_q[batch_index, cspace_index]
            X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(joint_axis, q))
        elif joint_type == wp.sim.JOINT_FIXED:
            X_jc = wp.transform_identity()
        # TODO: Add support for prismatic joints.
        #elif joint_type == wp.sim.JOINT_PRISMATIC:
        #    print("<encountered prismatic joints>")

        X_wc = X_wc * X_pj * X_jc
        
    link_transforms[batch_index, target_link_index] = X_wc


@wp.kernel
def eval_kinematics_with_velocities_and_axes_kernel(
        # inputs
        batch_cspace_q: wp.array(dtype=float, ndim=2),
        batch_cspace_qd: wp.array(dtype=float, ndim=2),
        num_links: wp.int32,
        joint_types: wp.array(dtype=int, ndim=1),
        joint_parents: wp.array(dtype=int, ndim=1),
        joint_transforms_in_parent_coords: wp.array(dtype=wp.transform, ndim=1),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        link2cspace: wp.array(dtype=int, ndim=1),
        # outputs
        batch_link_transforms: wp.array(dtype=wp.transform, ndim=2),
        batch_joint_axes: wp.array(dtype=wp.vec3, ndim=2),
        batch_link_spatial_velocities: wp.array(dtype=wp.spatial_vector, ndim=2)):

    batch_index = wp.tid()

    for link_index in range(num_links):
        parent_index = joint_parents[link_index]
        parent_origin = wp.vec3()
        parent_spatial_velocity = wp.spatial_vector()
        X_parent2world = wp.transform_identity()
        if parent_index >= 0:
            X_parent2world = batch_link_transforms[batch_index, parent_index]
            parent_origin = transform_get_translation(X_parent2world)
            parent_spatial_velocity = batch_link_spatial_velocities[batch_index, parent_index]

        local_joint_axis = local_joint_axes[link_index]
        X_joint2parent = joint_transforms_in_parent_coords[link_index]

        joint_type = joint_types[link_index]
        cspace_index = link2cspace[link_index]
        if joint_type == wp.sim.JOINT_REVOLUTE:
            q = batch_cspace_q[batch_index, cspace_index]
            qd = batch_cspace_qd[batch_index, cspace_index]  # Used in the conditional block below.

            X_child2joint = wp.transform(wp.vec3(), wp.quat_from_axis_angle(local_joint_axis, q))

        elif joint_type == wp.sim.JOINT_FIXED:
            X_child2joint = wp.transform_identity()
        
        # Compute world transform of link and extract the joint axis. Note that prev_link_transform is
        # an alias for X_parent2world
        X_child2world = X_parent2world * X_joint2parent * X_child2joint

        child_origin = transform_get_translation(X_child2world)
        parent_orig_to_child_orig = child_origin - parent_origin

        # This link's angular velocity is affected by the current joint. But note that the link origin's velocity
        # (which is also the joint origin) isn't affected by the current joint. So we want to compute the new
        # linear velocity first using the previous link's angular velocity.
        parent_angular_velocity = wp.spatial_top(parent_spatial_velocity)
        parent_linear_velocity = wp.spatial_bottom(parent_spatial_velocity)
        linear_velocity = parent_linear_velocity + wp.cross(parent_angular_velocity, parent_orig_to_child_orig)

        if joint_type == wp.sim.JOINT_REVOLUTE:
            # If this is a revolute joint, the angular velocity of the link will update and we'll shift
            # the origin to the new joint.
            joint_axis_in_world_coords = transform_vector(X_child2world, local_joint_axis)  # Compute joint axis in world coords.
            angular_velocity = parent_angular_velocity + qd * joint_axis_in_world_coords

            # Write the world coord joint axis to memory.
            batch_joint_axes[batch_index, cspace_index] = joint_axis_in_world_coords

        spatial_velocity = wp.spatial_vector(angular_velocity, linear_velocity)

        # Write the outputs to memory.
        batch_link_transforms[batch_index, link_index] = X_child2world
        batch_link_spatial_velocities[batch_index, link_index] = spatial_velocity


@wp.kernel
def eval_kinematics_with_velocities_and_axes_multithreaded_kernel(
        # inputs
        batch_cspace_q: wp.array(dtype=float, ndim=2),
        batch_cspace_qd: wp.array(dtype=float, ndim=2),
        joint_types: wp.array(dtype=int, ndim=1),
        joint_parents: wp.array(dtype=int, ndim=1),
        joint_transforms_in_parent_coords: wp.array(dtype=wp.transform, ndim=1),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        link_paths: wp.array(dtype=int, ndim=2),
        link2cspace: wp.array(dtype=int, ndim=1),
        # outputs
        batch_link_transforms: wp.array(dtype=wp.transform, ndim=2),
        batch_joint_axes: wp.array(dtype=wp.vec3, ndim=2),
        batch_link_spatial_velocities: wp.array(dtype=wp.spatial_vector, ndim=2)):

    batch_index, target_link_index = wp.tid()

    parent_origin = wp.vec3()
    parent_spatial_velocity = wp.spatial_vector()
    X_parent2world = wp.transform_identity()

    path_length = link_paths[target_link_index, 0]
    for i in range(path_length):
        link_index = link_paths[target_link_index, i+1]
        cspace_index = link2cspace[link_index]

        local_joint_axis = local_joint_axes[link_index]
        X_joint2parent = joint_transforms_in_parent_coords[link_index]

        joint_type = joint_types[link_index]
        if joint_type == wp.sim.JOINT_REVOLUTE:
            q = batch_cspace_q[batch_index, cspace_index]
            qd = batch_cspace_qd[batch_index, cspace_index]  # Used in the conditional block below.

            X_child2joint = wp.transform(wp.vec3(), wp.quat_from_axis_angle(local_joint_axis, q))

        elif joint_type == wp.sim.JOINT_FIXED:
            X_child2joint = wp.transform_identity()
        
        # Compute world transform of link and extract the joint axis. Note that prev_link_transform is
        # an alias for X_parent2world
        X_child2world = X_parent2world * X_joint2parent * X_child2joint

        child_origin = transform_get_translation(X_child2world)
        parent_orig_to_child_orig = child_origin - parent_origin

        # This link's angular velocity is affected by the current joint. But note that the link origin's velocity
        # (which is also the joint origin) isn't affected by the current joint. So we want to compute the new
        # linear velocity first using the previous link's angular velocity.
        parent_angular_velocity = wp.spatial_top(parent_spatial_velocity)
        parent_linear_velocity = wp.spatial_bottom(parent_spatial_velocity)
        linear_velocity = parent_linear_velocity + wp.cross(parent_angular_velocity, parent_orig_to_child_orig)

        if joint_type == wp.sim.JOINT_REVOLUTE:
            # If this is a revolute joint, the angular velocity of the link will update and we'll shift
            # the origin to the new joint.
            joint_axis_in_world_coords = transform_vector(X_child2world, local_joint_axis)  # Compute joint axis in world coords.
            angular_velocity = parent_angular_velocity + qd * joint_axis_in_world_coords

        spatial_velocity = wp.spatial_vector(angular_velocity, linear_velocity)

        # Setup variables for next cycle.
        X_parent2world = X_child2world
        parent_origin = child_origin
        parent_spatial_velocity = spatial_velocity

    # Write the outputs to memory.
    batch_link_transforms[batch_index, target_link_index] = X_child2world
    batch_link_spatial_velocities[batch_index, target_link_index] = spatial_velocity
    batch_joint_axes[batch_index, cspace_index] = joint_axis_in_world_coords


@wp.kernel
def joint_axes_kernel(
        # inputs
        cspace2link: wp.array(dtype=int, ndim=1),
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        # outputs
        link_joint_axes: wp.array(dtype=wp.vec3, ndim=2)):

    batch_i,joint_i = wp.tid()
    link_i = cspace2link[joint_i]
    world_axis = transform_vector(link_transforms[batch_i, link_i], local_joint_axes[link_i])
    link_joint_axes[batch_i, joint_i] = world_axis


@wp.kernel
def jacobians_from_axes_kernel(
        # inputs
        cspace2link: wp.array(dtype=int, ndim=1),
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        link_joint_axes: wp.array(dtype=wp.vec3, ndim=2),
        link_ancestory_matrix: wp.array(dtype=int, ndim=2),
        # outputs
        link_jacobians: wp.array(dtype=wp.vec3, ndim=3)):

    # link_i is the link we're taking the Jacobian for, and joint_link_i is the link corresponding
    # to the joint we're perturbing (the particular column of the Jacobian).
    batch_i,link_i,joint_i = wp.tid()
    joint_link_i = cspace2link[joint_i]

    if link_ancestory_matrix[joint_link_i, link_i] == 0:
        # The joint link isn't an ancestor of the link in question, so the Jacobian will be zero. We
        # can simply return here.
        return

    o = transform_get_translation(link_transforms[batch_i, joint_link_i])
    e = transform_get_translation(link_transforms[batch_i, link_i])
    p = e - o
    a = link_joint_axes[batch_i, joint_i]
    link_jacobians[batch_i, link_i, joint_i] = cross(a, p)


class KinematicsBase(object):
    """ Base class for kinematics implementations.
    
    Enables easily swapping between different underlying implementations.

    Note that object oriented abstractions don't affect performance since all python is ignored when
    creating Warp's cuda graphs. The graph is just the list of kernels, independent of the python
    interfacing structures used to organize the memory and launch the kernels.

    Forward kinematics (FK) here refers to the calculation of transforms and spacial velocities for
    each of the links. Additional, (using that information) we can calculate the Jacobians of the
    transforms as well.
    """
    def __init__(self, urdf_path, batch_size, device, verbose):
        self.device = device
        self.batch_size = batch_size

        setup_torch_to_use_warp_streams(verbose=verbose)

        if verbose:
            print("<parsing the URDF>")
        self.builder = wp.sim.ModelBuilder()

        if verbose:
            print("urdf_path:", urdf_path)
        self.urdf_info = parse_urdf(
                urdf_path, self.builder, wp.transform_identity(),
                include_collisions=False, verbose=verbose)
        self.model = self.builder.finalize(self.device)
        self.link_index_map = self.urdf_info.link_index_map
        self.link_names = [name for name in self.urdf_info.link_index_map]

        # Make some of the kinematics information that's not already in the model accessible from
        # the device.
        self.cspace2link = wp.array(self.urdf_info.cspace2link, dtype=int, device=self.device)
        self.link2cspace = wp.array(self.urdf_info.link2cspace, dtype=int, device=self.device)
        self.cspace_joint_limits = wp.array(self.urdf_info.cspace_joint_limits, dtype=float, device=self.device)

        # Aliases for kinematics info from model with more explicit names.
        self.joint_types = self.model.joint_type
        self.joint_parents = self.model.joint_parent
        self.joint_transforms_in_parent_coords = self.model.joint_X_p

        # NOTE: Warp > 0.7.2 made a change so that fixed joints are removed
        # from self.model.joint_axis. This breaks our kinematics kernels. The fix
        # is to reinject [0,0,0] (fixed joint axis) for every fixed joint at the right
        # location in the joint_axis array.

        # Get the total number of joints including fixed joints
        num_joints = self.joint_types.shape[0]

        # Allocate a 2D numpy array with 0s
        local_joint_axes_np = np.zeros((num_joints, 3))

        # Create numpy array for non-fixed joint axis data
        local_articulated_joint_axes_np = self.model.joint_axis.numpy()

        # Find where joint_types are not equal to 3 (3 indicates a fixed joint) 
        joint_types_np = self.joint_types.numpy()
        non_fixed_joint_indices = np.where(joint_types_np!=3)[0]

        # Inject non-fixed joint axis data into the full joint axes data array
        local_joint_axes_np[non_fixed_joint_indices] = local_articulated_joint_axes_np

        # Convert the numpy joint axis data to warp array
        self.local_joint_axes = wp.array(local_joint_axes_np, dtype=wp.vec3, device=self.device)

        self.cspace_dim = len(self.model.joint_q)
        self.num_links = len(self.model.joint_type)

        # Extract the link paths and dependencies.
        self.link_paths_numpy = extract_link_paths(self.joint_parents.numpy())
        self.link_paths = paths_to_warp(self.link_paths_numpy, device=self.device)
        self.link_ancestory_matrix = make_ancestory_matrix_from_paths(self.link_paths_numpy, device=self.device)

        # TODO: remove this section once the slowdown is debugged. It's currently identified to be in the 
        # torch solve. 
        #
        # Somehow having just the right amount of extra allocated memory makes fabrics run faster...
        # Doesn't affect the kinematics benchmarks, just the fabrics benchmark.
        extra_floats = 378000  # = 2*batch_size*self.num_links*self.cspace_dim*3 for batch 1000
        # The following bracket the separation point between fast and slow.
        #extra_floats = 336000  # fast
        #extra_floats = 335000  # slow
        #extra_floats = 100*335000  # large
        self.extra_memory = wp.zeros(shape=extra_floats, dtype=float, device=self.device)
                
    def get_cspace_index(self, joint_name):
        return self.urdf_info.cspace_name2index_map[joint_name]

    def get_link_index(self, link_name):
        return self.link_index_map[link_name]

    # TODO: update this interface
    def eval(self, batch_q, batch_qd, velocities, jacobians):
        """ Primary interface to evaluation that deriving classes should implement.

        Evaluation should support all combinations of true / false values for velocities
        and jacobians. If both are false, then by default it should just compute the forward
        kinematic link transforms. If velocities is true, then it should additionally compute
        the link spatial velocities (angular and linear). If jacobians is true, it should 
        compute both positional and rotational jacobians for each link. Note that rotational 
        jacobians are often just represented as a collection of joint axes in root coordinates
        (the exponential map Jacobian).
        """
        raise NotImplementedError()


class Kinematics(KinematicsBase):
    def __init__(self, urdf_path, batch_size, thread_across_links=True, device="cuda", verbose=False):
        super().__init__(urdf_path, batch_size, device, verbose)

        self.thread_across_links = thread_across_links
        self.active_batch_size = None

        # Output space
        self.batch_link_transforms = wp.zeros(
                shape=(batch_size, self.num_links), dtype=wp.transform, device=self.device,
                requires_grad=True)
        self.batch_link_spatial_velocities = wp.zeros(
                shape=(batch_size, self.num_links), dtype=wp.spatial_vector, device=self.device,
                requires_grad=True)
        self.batch_joint_axes = wp.zeros(
                shape=(batch_size, self.cspace_dim), dtype=wp.vec3, device=self.device,
                requires_grad=True)
        self.batch_link_jacobians = wp.zeros(
                shape=(batch_size, self.num_links, self.cspace_dim), dtype=wp.vec3,
                device=self.device,
                requires_grad=True)
        
        # Torch handles
        self.batch_link_transforms_torch = wp.torch.to_torch(self.batch_link_transforms)
        self.batch_link_spatial_velocities_torch = wp.torch.to_torch(self.batch_link_spatial_velocities)
        self.batch_joint_axes_torch = wp.torch.to_torch(self.batch_joint_axes)
        self.batch_link_jacobians_torch = wp.torch.to_torch(self.batch_link_jacobians)

    def eval(self, batch_q, batch_qd=None, jacobians=False, thread_across_links=None):
        """ Evaluate the kinematics given the provide information.

        There are a number of different configurations for this eval.
        1. Transforms only kinematics: eval(batch_q)
        2. Transforms with spatial velocities eval(batch_q, batch_qd)
        3. Include Jacobians: eval(..., jacobians=True)

        Additionally, on construction the Kinematics object's thread_across_links can be
        set to True to multithread the computations across the links. That object-level attribute can
        be overridded per call to eval() using:

        4. Multithread across links eval(..., thread_across_links=True)
        """
        self.active_batch_size = min(batch_q.shape[0], self.batch_size)
        if thread_across_links is None:
            thread_across_links = self.thread_across_links
        if batch_qd is not None:
            if thread_across_links:
                self._fk_transforms_with_spatial_velocities_multithreaded(batch_q, batch_qd)
            else:
                self._fk_transforms_with_spatial_velocities(batch_q, batch_qd)
        else:
            if thread_across_links:
                self._fk_transforms_multithreaded(batch_q)
            else:
                self._fk_transforms(batch_q)

            if jacobians:
                self._axes_from_fk_transforms()

        if jacobians:
            self._jacobians_from_axes()

    def fd_position_jacobian_torch(self, q_torch, link_index, 
            eps=1e-4, 
            thread_across_links=None):
        d = len(q_torch)  # Dimension of the C-space
        if d != self.cspace_dim:
            raise RuntimeError("q_torch has the wrong number of dimensions. Found {}, should be {}".format(
                d, self.cspace_dim))

        # Construct a batch of configs for computing the fd Jacobian in parallel. The first d batch
        # configurations are perturbed, and final d+1st config is the original config.
        batch_q_torch = q_torch.repeat(d+1, 1)
        batch_q_torch[:d,:] += eps * torch.eye(d)

        # Compute the FK for the batch and extract the positions.
        self.eval(wp.torch.from_torch(batch_q_torch), thread_across_links=thread_across_links)
        X = self.batch_link_transforms_torch[:(d+1), link_index, :3]

        # Compute the Jacobian transpose using finite-differencing of the perturbed positions.
        Jt = (X[:-1,:] - X[-1,:].repeat(d,1)) / eps
        return Jt.t()  # Return the transpose of that.

    def _fk_transforms(self, batch_q):
        wp.launch(kernel=link_transforms_kernel,
                  dim=self.active_batch_size,
                  inputs=[
                      batch_q,
                      self.num_links,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link2cspace],
                  outputs=[
                      self.batch_link_transforms],
                  device=self.device)

    def _fk_transforms_multithreaded(self, batch_q):
        wp.launch(kernel=link_transforms_multithreaded_kernel,
                  dim=(self.active_batch_size, self.num_links),
                  inputs=[
                      batch_q,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link_paths,
                      self.link2cspace,
                      self.batch_link_transforms],
                  device=self.device)

    def _fk_transforms_with_spatial_velocities(self, batch_q, batch_qd):
        wp.launch(kernel=eval_kinematics_with_velocities_and_axes_kernel,
                  dim=self.active_batch_size,
                  inputs=[
                      batch_q,
                      batch_qd,
                      self.joint_types.shape[0],  # num links
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link2cspace,
                      self.batch_link_transforms,
                      self.batch_joint_axes,
                      self.batch_link_spatial_velocities],
                  device=self.device)
        return self.batch_link_transforms, self.batch_link_spatial_velocities

    def _fk_transforms_with_spatial_velocities_multithreaded(self, batch_q, batch_qd):
        wp.launch(kernel=eval_kinematics_with_velocities_and_axes_multithreaded_kernel,
                  dim=(self.active_batch_size, self.num_links),
                  inputs=[
                      batch_q,
                      batch_qd,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link_paths,
                      self.link2cspace,
                      self.batch_link_transforms,
                      self.batch_joint_axes,
                      self.batch_link_spatial_velocities],
                  device=self.device)
        return self.batch_link_transforms, self.batch_link_spatial_velocities

    def _axes_from_fk_transforms(self):
        wp.launch(kernel=joint_axes_kernel,
                  dim=(self.active_batch_size, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.local_joint_axes],
                  outputs=[self.batch_joint_axes], 
                  device=self.device)

    def _jacobians_from_axes(self):
        wp.launch(kernel=jacobians_from_axes_kernel,
                  dim=(self.active_batch_size, self.num_links, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.batch_joint_axes,
                          self.link_ancestory_matrix],
                  outputs=[self.batch_link_jacobians], 
                  device=self.device)


#======================================================================================
# Everything below is legacy, retained for benchmarking purposes
#======================================================================================


@wp.kernel
def eval_kinematics_with_jacobians_kernel(
        cspace_q: wp.array(dtype=float, ndim=2),
        num_jacobians_needed: int,
        jacobian_link_indices: wp.array(dtype=int, ndim=1),
        cspace2link: wp.array(dtype=int, ndim=1),
        num_links: wp.int32,
        joint_types: wp.array(dtype=int, ndim=1),
        joint_parents: wp.array(dtype=int, ndim=1),
        joint_transforms_in_parent_coords: wp.array(dtype=wp.transform, ndim=1),
        local_joint_axes: wp.array(dtype=wp.vec3, ndim=1),
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        #link_jacobians: wp.array(dtype=wp.vec3, ndim=3)):
        link_jacobians: wp.array(dtype=wp.spatial_vector, ndim=3)):

    tid = wp.tid()

    cspace_index = int(0)
    for link_index in range(num_links):

        parent_index = joint_parents[link_index]
        X_wp = wp.transform_identity()
        if (parent_index >= 0):
            X_wp = link_transforms[tid, parent_index]

        joint_axis = local_joint_axes[link_index]
        X_pj = joint_transforms_in_parent_coords[link_index]
        
        joint_type = joint_types[link_index]
        if joint_type == wp.sim.JOINT_REVOLUTE:
            q = cspace_q[tid, cspace_index]
            cspace_index += 1
            X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(joint_axis, q))
        elif joint_type == wp.sim.JOINT_FIXED:
            X_jc = wp.transform_identity()

        link_transforms[tid, link_index] = X_wp * X_pj * X_jc

    cspace_dim = cspace_index
    jacobian_i = int(0)
    for jacobian_i in range(num_jacobians_needed):
        link_i = jacobian_link_indices[jacobian_i]
        cspace_i = int(0)
        finished = int(0)
        while finished == 0:
            joint_link_i = cspace2link[cspace_i]
            if cspace_i == cspace_dim or joint_link_i >= link_i:
                finished = 1
            else:
                a = transform_vector(link_transforms[tid, joint_link_i], local_joint_axes[joint_link_i])
                o = transform_get_translation(link_transforms[tid, joint_link_i])
                e = transform_get_translation(link_transforms[tid, link_i])
                p = e - o
                link_jacobians[tid, jacobian_i, cspace_i] = wp.spatial_vector(cross(a, p), a)
                #link_jacobians[tid, jacobian_i, cspace_i] = cross(p, a)

                cspace_i += 1


@wp.kernel
def task_space_velocities_from_jacobians_kernel(
        # Inputs
        batch_qd: wp.array(dtype=float, ndim=2),
        cspace_dim: int,
        batch_link_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        batch_link_origin_velocities: wp.array(dtype=wp.vec3, ndim=2)):
        
    batch_index, link_index = wp.tid()

    qd = batch_qd[batch_index]
    J = batch_link_jacobians[batch_index, link_index]
    v = vec3()
    for i in range(cspace_dim):
        v += J[i] * qd[i]
    batch_link_origin_velocities[batch_index, link_index] = v


@wp.func
def quat_to_matrix_direct(quat: wp.quat):
    # ref: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qw = quat[3]
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    R = wp.mat33(
            1. - 2.*qy2 - 2.*qz2, 2.*qx*qy - 2.*qz*qw, 2.*qx*qz + 2.*qy*qw,
            2.*qx*qy + 2.*qz*qw, 1. - 2.*qx2 - 2.*qz2, 2.*qy*qz - 2.*qx*qw,
            2.*qx*qz - 2.*qy*qw, 2.*qy*qz + 2.*qx*qw, 1. - 2.*qx2 - 2.*qy2)
    return R


@wp.kernel
def calc_rotation_matrices_kernel(
        # inputs
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        # outputs
        link_rotation_matrices: wp.array(dtype=wp.mat33, ndim=2)):

    batch_i, link_i = wp.tid()
    quat = transform_get_rotation(link_transforms[batch_i, link_i])
    link_rotation_matrices[batch_i, link_i] = quat_to_matrix_direct(quat)
    #world_axis = transform_vector(link_transforms[batch_i, link_i], local_joint_axes[link_i])
    #link_joint_axes[batch_i, joint_i] = world_axis


@wp.kernel
def calc_frame_jacobians_kernel(
        # inputs
        cspace2link: wp.array(dtype=int, ndim=1),
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        link_rotation_matrices: wp.array(dtype=wp.mat33, ndim=2),
        link_joint_axes: wp.array(dtype=wp.vec3, ndim=2),
        # outputs
        link_frame_jacobians: wp.array(dtype=wp.vec3, ndim=4)):
        #link_orig_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        #link_axis_x_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        #link_axis_y_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        #link_axis_z_jacobians: wp.array(dtype=wp.vec3, ndim=3)):

    # link_i is the link we're taking the Jacobian for, and joint_link_i is the link corresponding
    # to the joint we're perturbing (the particular column of the Jacobian).
    batch_i,link_i,link_element_i, joint_i = wp.tid()
    joint_link_i = cspace2link[joint_i]

    if joint_link_i > link_i:
        # All joints after a link don't affect the link's frame.
        return
    elif link_element_i == 3 and joint_link_i == link_i:
        # A given link's joint doesn't affect the link's origin.
        return

    a = link_joint_axes[batch_i, joint_i]
    if link_element_i < 3:
        # TODO: is there a better way of extracting the column?
        r = vec3(link_rotation_matrices[batch_i, link_i][link_element_i, 0],
                 link_rotation_matrices[batch_i, link_i][link_element_i, 1],
                 link_rotation_matrices[batch_i, link_i][link_element_i, 2])
        c = cross(r, a)
    else:
        o = transform_get_translation(link_transforms[batch_i, joint_link_i])
        e = transform_get_translation(link_transforms[batch_i, link_i])
        p = e - o
        c = cross(p, a)

    link_frame_jacobians[batch_i, link_i, link_element_i, joint_i] = c


@wp.kernel
def calc_frame_jacobians_kernel2(
        # inputs
        cspace2link: wp.array(dtype=int, ndim=1),
        link_transforms: wp.array(dtype=wp.transform, ndim=2),
        link_rotation_matrices: wp.array(dtype=wp.mat33, ndim=2),
        link_joint_axes: wp.array(dtype=wp.vec3, ndim=2),
        # outputs
        link_orig_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        link_axis_x_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        link_axis_y_jacobians: wp.array(dtype=wp.vec3, ndim=3),
        link_axis_z_jacobians: wp.array(dtype=wp.vec3, ndim=3)):

    # link_i is the link we're taking the Jacobian for, and joint_link_i is the link corresponding
    # to the joint we're perturbing (the particular column of the Jacobian).
    batch_i,link_i,link_element_i, joint_i = wp.tid()
    joint_link_i = cspace2link[joint_i]

    if joint_link_i > link_i:
        # All joints after a link don't affect the link's frame.
        return
    elif link_element_i == 3 and joint_link_i == link_i:
        # A given link's joint doesn't affect the link's origin.
        return

    a = link_joint_axes[batch_i, joint_i]
    if link_element_i < 3:
        # TODO: is there a better way of extracting the column?
        r = vec3(link_rotation_matrices[batch_i, link_i][link_element_i, 0],
                 link_rotation_matrices[batch_i, link_i][link_element_i, 1],
                 link_rotation_matrices[batch_i, link_i][link_element_i, 2])
        c = cross(r, a)
    else:
        o = transform_get_translation(link_transforms[batch_i, joint_link_i])
        e = transform_get_translation(link_transforms[batch_i, link_i])
        p = e - o
        c = cross(p, a)

    if link_element_i == 0:
        link_axis_x_jacobians[batch_i, link_i, joint_i] = c
    elif link_element_i == 1:
        link_axis_y_jacobians[batch_i, link_i, joint_i] = c
    elif link_element_i == 2:
        link_axis_z_jacobians[batch_i, link_i, joint_i] = c
    elif link_element_i == 3:
        link_orig_jacobians[batch_i, link_i, joint_i] = c


class KinematicsLegacy(object):
    """ Legacy kinematics object.

    Multiple implementations are packed into the same kinematics object. The objects deriving from
    KinematicsBase below are better separated. New kinematics implementations should derive from
    that hierarchy.

    Note that object oriented abstractions don't affect performance since all python is ignored when
    creating Warp's cuda graphs. The graph is just the list of kernels, independent of the python
    interfacing structures used to organize the memory and launch the kernels.
    """
    def __init__(self, batch_size, model, link_index_map, cspace2link, cspace_joint_limits,
            jacobians_needed):
        self.device = model.device
        self.batch_size = batch_size
        self.model = model
        self.link_index_map = link_index_map

        # Aliases for kinematics info from model with more explicit names.
        self.cspace2link = wp.array(cspace2link, dtype=int, device=self.device)
        self.cspace_joint_limits = wp.array(cspace_joint_limits, dtype=float, device=self.device)

        self.joint_types = model.joint_type
        self.joint_parents = model.joint_parent
        self.joint_transforms_in_parent_coords = model.joint_X_p
        self.local_joint_axes = model.joint_axis

        self.cspace_dim = len(model.joint_q)
        self.num_links = len(model.joint_type)

        #self.jacobians_needed = ["right_gripper", "right_gripper_x", "right_gripper_z"]
        #self.jacobians_needed = ["right_gripper", "right_gripper_x"]
        #self.jacobians_needed = ["right_gripper"]
        self.jacobians_needed = jacobians_needed
        self.jacobian_link_indices = wp.array([
                self.link_index_map[link_name] for link_name in self.jacobians_needed],
                dtype=int, device=self.device)

        # Output space
        self.batch_link_transforms = wp.zeros(shape=(batch_size, self.num_links), dtype=wp.transform,
                device=self.device)
        self.link_transforms = self.batch_link_transforms  # legacy naming
        self.batch_link_spatial_velocities = wp.zeros(shape=(batch_size, self.num_links),
                dtype=wp.spatial_vector, device=self.device)
        self.batch_joint_axes = wp.zeros(shape=(batch_size, self.cspace_dim), dtype=wp.vec3,
                device=self.device)
        self.link_joint_axes = self.batch_joint_axes  # Legacy naming
        self.link_rotation_matrices = wp.zeros(shape=(batch_size, self.num_links), dtype=wp.mat33, 
                device=self.device)
        self.batch_link_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_jacobians = self.batch_link_jacobians  # Legacy naming
        self.link_jacobians_full = wp.zeros(shape=(batch_size, len(self.jacobians_needed), self.cspace_dim),
                dtype=wp.spatial_vector, device=self.device)
        self.link_frame_jacobians = wp.zeros(shape=(batch_size, self.num_links, 4, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_orig_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_x_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_y_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_z_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.batch_link_origin_velocities = wp.zeros(
                shape=(batch_size, self.num_links),
                dtype=wp.vec3,
                device=self.device)

        # Extract the link paths.
        self.link_paths = paths_to_warp(extract_link_paths(self.joint_parents.numpy()), device=self.device)

    def get_link_index(self, link_name):
        return self.link_index_map[link_name]

    def eval(self, cspace_q):
        wp.launch(kernel=link_transforms_kernel,
                  dim=self.batch_size,
                  inputs=[    
                      cspace_q,
                      self.joint_types.shape[0],  # num links
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes],
                  outputs=[self.link_transforms],
                  device=self.device)
        return self.link_transforms

    def eval_with_axes(self, cspace_q):
        self.eval(cspace_q)  # Calculates self.link_transforms
        wp.launch(kernel=joint_axes_kernel,
                  dim=(self.batch_size, self.cspace_dim),
                  inputs=[self.cspace2link, self.link_transforms, self.local_joint_axes, self.batch_joint_axes], 
                  device=self.device)
        return self.link_transforms, self.link_joint_axes

    # TODO: switch naming conventions to batch_q, etc.
    def eval_with_jacobians(self, cspace_q):
        self.eval_with_axes(cspace_q)
        wp.launch(kernel=jacobians_from_axes_kernel,
                  dim=(self.batch_size, self.num_links, self.cspace_dim),
                  inputs=[self.cspace2link, self.link_transforms, self.link_joint_axes], 
                  outputs=[self.link_jacobians],
                  device=self.device)
        return self.link_transforms, self.link_jacobians

    def eval_kinematics_with_jacobians(self, cspace_q):
        wp.launch(kernel=eval_kinematics_with_jacobians_kernel,
                  dim=self.batch_size,
                  inputs=[    
                      cspace_q,
                      len(self.jacobian_link_indices),
                      self.jacobian_link_indices,
                      self.cspace2link,
                      self.joint_types.shape[0],  # num links
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link_transforms,
                      #self.link_jacobians],
                      self.link_jacobians_full],
                  device=self.device)
        return self.link_transforms, self.link_jacobians_full

    def eval_with_rotation_matrices(self, cspace_q):
        # TODO: This could probably be more efficient. Here we're calculating the joint axes first,
        # then calculating the full rotation matrices for each link. We can extract the axis
        # information from the rotation matrices, or assume the DH convention. Although, might not
        # matter all that much at the end of the day. Something to experiment with.
        self.eval_with_axes(cspace_q)
        wp.launch(kernel=calc_rotation_matrices_kernel,
                  dim=(self.batch_size, self.num_links),
                  inputs=[self.link_transforms], 
                  outputs=[self.link_rotation_matrices],
                  device=self.device)
        return self.link_transforms, self.link_rotation_matrices

    def eval_with_frame_jacobians(self, cspace_q):
        self.eval_with_rotation_matrices(cspace_q)
        wp.launch(kernel=calc_frame_jacobians_kernel,
                  dim=(self.batch_size, self.num_links, 4, self.cspace_dim),
                  inputs=[self.cspace2link, self.link_transforms, self.link_rotation_matrices, self.link_joint_axes], 
                  outputs=[self.link_frame_jacobians],
                  device=self.device)
        return self.link_transforms, self.link_frame_jacobians

    def eval_with_frame_jacobians2(self, cspace_q):
        self.eval_with_rotation_matrices(cspace_q)
        wp.launch(kernel=calc_frame_jacobians_kernel2,
                  dim=(self.batch_size, self.num_links, 4, self.cspace_dim),
                  inputs=[self.cspace2link, self.link_transforms, self.link_rotation_matrices, self.link_joint_axes], 
                  outputs=[
                      self.link_orig_jacobians,
                      self.link_axis_x_jacobians,
                      self.link_axis_y_jacobians,
                      self.link_axis_z_jacobians],
                  device=self.device)
        return self.link_transforms, self.link_frame_jacobians

    def eval_fk(self, batch_cspace_q, batch_cspace_qd, thread_across_links=False):
        if thread_across_links:
            wp.launch(kernel=eval_kinematics_with_velocities_and_axes_multithreaded_kernel,
                      dim=(self.batch_size, self.num_links),
                      inputs=[    
                          batch_cspace_q,
                          batch_cspace_qd,
                          self.joint_types,
                          self.joint_parents,
                          self.joint_transforms_in_parent_coords,
                          self.local_joint_axes,
                          self.link_paths,
                          self.batch_link_transforms,
                          self.batch_joint_axes,
                          self.batch_link_spatial_velocities],
                      device=self.device)
        else:
            wp.launch(kernel=eval_kinematics_with_velocities_and_axes_kernel,
                      dim=self.batch_size,
                      inputs=[    
                          batch_cspace_q,
                          batch_cspace_qd,
                          self.joint_types.shape[0],  # num links
                          self.joint_types,
                          self.joint_parents,
                          self.joint_transforms_in_parent_coords,
                          self.local_joint_axes,
                          self.batch_link_transforms,
                          self.batch_joint_axes,
                          self.batch_link_spatial_velocities],
                      device=self.device)

        return self.batch_link_transforms, self.batch_link_spatial_velocities

    def eval_fk_with_jacobians(self, batch_cspace_q, batch_cspace_qd, thread_across_links=False):
        self.eval_fk(batch_cspace_q, batch_cspace_qd, thread_across_links)
        wp.launch(kernel=jacobians_from_axes_kernel,
                  dim=(self.batch_size, self.num_links, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.batch_joint_axes, self.batch_link_jacobians], 
                  device=self.device)
        return self.batch_link_transforms, self.batch_link_spatial_velocities, self.batch_link_jacobians

    def task_space_velocities(self, batch_qd):
        """ Calculates each link origin's linear velocity.
        
        Assumes the Jacobians have already been calculated for each of the links. 
        """
        wp.launch(kernel=task_space_velocities_from_jacobians_kernel,
                  dim=(self.batch_size, self.num_links),
                  inputs=[
                      batch_qd,
                      self.cspace_dim,
                      self.batch_link_jacobians,
                      self.batch_link_origin_velocities],
                  device=self.device)


class KinematicsNoVelocities(KinematicsBase):
    """ Implements the portion of the kinematics API that doesn't include velocities.
    """
    def __init__(self, urdf_path, batch_size, thread_across_links=True, device="cuda", verbose=False):
        super().__init__(urdf_path, batch_size, device, verbose)

        self.thread_across_links = thread_across_links

        # Output space
        self.batch_link_transforms = wp.zeros(shape=(batch_size, self.num_links), dtype=wp.transform,
                device=self.device)
        self.batch_joint_axes = wp.zeros(shape=(batch_size, self.cspace_dim), dtype=wp.vec3,
                device=self.device)
        self.batch_link_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)

    def eval_link_transforms(self, batch_q):
        if self.thread_across_links:
            self._fk_transforms_multithreaded(batch_q)
        else:
            self._fk_transforms(batch_q)
        return self.batch_link_transforms

    def eval_link_transforms_with_jacobians(self, batch_q):
        self.eval_link_transforms(batch_q)
        self._axes_from_fk_transforms()
        self._jacobians_from_axes()

    def _fk_transforms(self, batch_q):
        wp.launch(kernel=link_transforms_kernel,
                  dim=self.batch_size,
                  inputs=[    
                      batch_q,
                      self.num_links,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.batch_link_transforms],
                  device=self.device)

    def _fk_transforms_multithreaded(self, batch_q):
        wp.launch(kernel=link_transforms_multithreaded_kernel,
                  dim=(self.batch_size, self.num_links),
                  inputs=[    
                      batch_q,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link_paths,
                      self.batch_link_transforms],
                  device=self.device)

    def _axes_from_fk_transforms(self):
        wp.launch(kernel=joint_axes_kernel,
                  dim=(self.batch_size, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.local_joint_axes, self.batch_joint_axes], 
                  device=self.device)

    def _jacobians_from_axes(self):
        wp.launch(kernel=jacobians_from_axes_kernel,
                  dim=(self.batch_size, self.num_links, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.batch_joint_axes, self.batch_link_jacobians], 
                  device=self.device)

class KinematicsStagewise(KinematicsNoVelocities):
    """ A stagewise implementation of kinematics using separate kernels to compute many incremental
    parts.

    Stages:
    - forward pass link transforms (optionally multithreaded)
    - compute joint axes from link transforms (not this is the rotational Jacobian)
    - compute jacobians from joint axes and link tranforms
    - calculates link origin velocities from Jacobians xd = J qd

    Note: Should calculate spatial velocities from Jacobians using xd = J qd (using the axes as the
    rotational Jacobian).
    """
    def __init__(self, urdf_path, batch_size, thread_across_links=True, device="cuda", verbose=False):
        super().__init__(urdf_path, batch_size, thread_across_links, device, verbose)

        # Output space
        self.batch_link_origin_velocities = wp.zeros(
                shape=(batch_size, self.num_links),
                dtype=wp.vec3,
                device=self.device)

    def eval_link_transforms_with_spatial_velocities(self, batch_q, batch_qd):
        """ Warning: currently this method returns link origin velocities rather than spatial
        velocites.
        """
        self.eval_link_transforms_with_jacobians(batch_q)
        self._task_space_velocities_from_jacobians(batch_qd)
        return self.batch_link_transforms, self.batch_link_origin_velocities

    def eval_link_transforms_with_spatial_velocities_and_jacobians(self, batch_q, batch_qd):
        # Jacobians are computed as a substep of calculating the velocities.
        self.eval_link_transforms_with_spatial_velocities(batch_q, batch_qd)
        return self.batch_link_transforms, self.batch_link_origin_velocities, self.batch_link_jacobians

    def _task_space_velocities_from_jacobians(self, batch_qd):
        """ Calculates each link origin's linear velocity.
        
        Assumes the Jacobians have already been calculated for each of the links. 
        """
        wp.launch(kernel=task_space_velocities_from_jacobians_kernel,
                  dim=(self.batch_size, self.num_links),
                  inputs=[
                      batch_qd,
                      self.cspace_dim,
                      self.batch_link_jacobians,
                      self.batch_link_origin_velocities],
                  device=self.device)


class KinematicsOrigInterface(KinematicsNoVelocities):
    def __init__(self, urdf_path, batch_size, thread_across_links=True, device="cuda", verbose=False):
        super().__init__(urdf_path, batch_size, thread_across_links, device, verbose)

        # Output space
        self.batch_link_transforms = wp.zeros(shape=(batch_size, self.num_links), dtype=wp.transform,
                device=self.device)
        self.batch_link_spatial_velocities = wp.zeros(shape=(batch_size, self.num_links),
                dtype=wp.spatial_vector, device=self.device)
        self.batch_joint_axes = wp.zeros(shape=(batch_size, self.cspace_dim), dtype=wp.vec3,
                device=self.device)
        self.batch_link_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)

    def eval_link_transforms_with_spatial_velocities(self, batch_q, batch_qd):
        if self.thread_across_links:
            return self._fk_transforms_with_spatial_velocities_multithreaded(batch_q, batch_qd)
        else:
            return self._fk_transforms_with_spatial_velocities(batch_q, batch_qd)

    def eval_link_transforms_with_spatial_velocities_and_jacobians(self, batch_q, batch_qd):
        self.eval_link_transforms_with_spatial_velocities(batch_q, batch_qd)
        self._jacobians_from_axes()
        return self.batch_link_transforms, self.batch_link_spatial_velocities, self.batch_link_jacobians

    def eval_fk_with_jacobians(self, batch_q, batch_qd):
        return self.eval_link_transforms_with_spatial_velocities_and_jacobians(batch_q, batch_qd)

    def _fk_transforms_with_spatial_velocities_multithreaded(self, batch_q, batch_qd):
        wp.launch(kernel=eval_kinematics_with_velocities_and_axes_multithreaded_kernel,
                  dim=(self.batch_size, self.num_links),
                  inputs=[    
                      batch_q,
                      batch_qd,
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.link_paths,
                      self.batch_link_transforms,
                      self.batch_joint_axes,
                      self.batch_link_spatial_velocities],
                  device=self.device)
        return self.batch_link_transforms, self.batch_link_spatial_velocities

    def _fk_transforms_with_spatial_velocities(self, batch_q, batch_qd):
        wp.launch(kernel=eval_kinematics_with_velocities_and_axes_kernel,
                  dim=self.batch_size,
                  inputs=[    
                      batch_q,
                      batch_qd,
                      self.joint_types.shape[0],  # num links
                      self.joint_types,
                      self.joint_parents,
                      self.joint_transforms_in_parent_coords,
                      self.local_joint_axes,
                      self.batch_link_transforms,
                      self.batch_joint_axes,
                      self.batch_link_spatial_velocities],
                  device=self.device)
        return self.batch_link_transforms, self.batch_link_spatial_velocities


class KinematicsRaw:
    def __init__(self, urdf_path, batch_size, thread_across_links, device, verbose=False):
        self.device = device
        self.batch_size = batch_size
        self.thread_across_links = thread_across_links

        if verbose:
            print("<parsing the URDF>")
        builder = wp.sim.ModelBuilder()

        if verbose:
            print("urdf_path:", urdf_path)
        link_index_map, cspace2link, cspace_joint_limits = parse_urdf(
                urdf_path, builder, wp.transform_identity(),
                include_collisions=False, verbose=verbose)
        self.link_index_map = link_index_map
        self.cspace2link = cspace2link
        self.cspace_joint_limits = cspace_joint_limits

        self.model = builder.finalize(self.device)
        self.link_index_map = link_index_map

        # Make some of the kinematics information that's not already in the model accessible from
        # the device.
        self.cspace2link = wp.array(cspace2link, dtype=int, device=self.device)
        self.cspace_joint_limits = wp.array(cspace_joint_limits, dtype=float, device=self.device)

        # Aliases for kinematics info from model with more explicit names.
        self.joint_types = self.model.joint_type
        self.joint_parents = self.model.joint_parent
        self.joint_transforms_in_parent_coords = self.model.joint_X_p
        self.local_joint_axes = self.model.joint_axis

        self.cspace_dim = len(self.model.joint_q)
        self.num_links = len(self.model.joint_type)

        # Extract the link paths.
        self.link_paths = paths_to_warp(extract_link_paths(self.joint_parents.numpy()), device=self.device)

        # Output space.
        self.batch_link_transforms = wp.zeros(shape=(batch_size, self.num_links), dtype=wp.transform,
                device=self.device)
        self.batch_link_spatial_velocities = wp.zeros(shape=(batch_size, self.num_links),
                dtype=wp.spatial_vector, device=self.device)
        self.batch_joint_axes = wp.zeros(shape=(batch_size, self.cspace_dim), dtype=wp.vec3,
                device=self.device)
        self.batch_link_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)

        # Somehow having this extra output space makes fabrics run faster... Doesn't affect the
        # kinematics benchmarks. We probably need more stable fabrics benchmarks.
        self.link_orig_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_x_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_y_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)
        self.link_axis_z_jacobians = wp.zeros(shape=(batch_size, self.num_links, self.cspace_dim),
                dtype=wp.vec3, device=self.device)

    def get_link_index(self, link_name):
        return self.link_index_map[link_name]

    def eval_fk_with_jacobians(self, batch_q, batch_qd):
        if self.thread_across_links:
            wp.launch(kernel=eval_kinematics_with_velocities_and_axes_multithreaded_kernel,
                      dim=(self.batch_size, self.num_links),
                      inputs=[    
                          batch_q,
                          batch_qd,
                          self.joint_types,
                          self.joint_parents,
                          self.joint_transforms_in_parent_coords,
                          self.local_joint_axes,
                          self.link_paths,
                          self.batch_link_transforms,
                          self.batch_joint_axes,
                          self.batch_link_spatial_velocities],
                      device=self.device)
        else:
            wp.launch(kernel=eval_kinematics_with_velocities_and_axes_kernel,
                      dim=self.batch_size,
                      inputs=[    
                          batch_q,
                          batch_qd,
                          self.joint_types.shape[0],  # num links
                          self.joint_types,
                          self.joint_parents,
                          self.joint_transforms_in_parent_coords,
                          self.local_joint_axes,
                          self.batch_link_transforms,
                          self.batch_joint_axes,
                          self.batch_link_spatial_velocities],
                      device=self.device)
        wp.launch(kernel=jacobians_from_axes_kernel,
                  dim=(self.batch_size, self.num_links, self.cspace_dim),
                  inputs=[self.cspace2link, self.batch_link_transforms, self.batch_joint_axes, self.batch_link_jacobians], 
                  device=self.device)

        return self.batch_link_transforms, self.batch_link_spatial_velocities, self.batch_link_jacobians

