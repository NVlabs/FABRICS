# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import torch
import warp as wp

@wp.kernel
def update_voxel_count(
    # inputs
    voxel_size: float,
    x_min: float,
    y_min: float,
    z_min: float,
    points: wp.array(dtype=wp.vec3, ndim=1),
    voxel_counter: wp.array(dtype=wp.int32, ndim=4),
    # outputs
    voxel_count: wp.array(dtype=wp.int32, ndim=1),
    voxel_centers: wp.array(dtype=wp.float32, ndim=2)):
    """
    Updates the counter for a voxel within the voxel grid given the query point and voxel
    settings.
    ----------------------------------------------
    :param voxel_size: float, length of voxel (assumed cube) dimension in meters
    :param x_min: float, the minimum x position for the voxel grid
    :param y_min: float, the minimum y position for the voxel grid
    :param z_min: float, the minimum z position for the voxel grid
    :param points: b dimensional warp array of 3D points (b = batch size)
    :param voxel_counter: bxixjxk warp array of ints, where int value is the voxel visitation count
                          b = batch size
                          i = number of voxels along x
                          j = number of voxels along y
                          k = number of voxels along z
    :return voxel_count: b dim warp array of ints capturing the voxel visition count
                         for the voxel belonging to the query point in points
    :return voxel_centers: (bx3) 2D warp array of floats that contain the 3D voxel center
                           of the voxel that associates with the query point
    """
    batch_index, voxel_x_index, voxel_y_index, voxel_z_index = wp.tid()

    # The 3D point for which we want to find the associated voxel
    query_point = points[batch_index]

    # Find the upper and lower x,y,z limits for the current voxel
    x_lower = x_min + float(voxel_x_index) * voxel_size
    x_upper = x_min + (float(voxel_x_index) + 1.) * voxel_size
    y_lower = y_min + float(voxel_y_index) * voxel_size
    y_upper = y_min + (float(voxel_y_index) + 1.) * voxel_size
    z_lower = z_min + float(voxel_z_index) * voxel_size
    z_upper = z_min + (float(voxel_z_index) + 1.) * voxel_size

    # Check to see if the query point falls within the current voxel
    if query_point[0] >= x_lower and query_point[0] < x_upper:
        if query_point[1] >= y_lower and query_point[1] < y_upper:
            if query_point[2] >= z_lower and query_point[2] < z_upper:
                # Increment the count in the voxel grid
                voxel_counter[batch_index, voxel_x_index, voxel_y_index, voxel_z_index] = \
                    voxel_counter[batch_index, voxel_x_index, voxel_y_index, voxel_z_index] + 1
                # Return the count for the associated voxel for this query point
                voxel_count[batch_index] =\
                    voxel_counter[batch_index, voxel_x_index, voxel_y_index, voxel_z_index]

                # Calculate voxel center
                voxel_centers[batch_index, 0] = (x_upper + x_lower) / 2.
                voxel_centers[batch_index, 1] = (y_upper + y_lower) / 2.
                voxel_centers[batch_index, 2] = (z_upper + z_lower) / 2.

class VoxelCounter(object):
    def __init__(self, batch_size, device, voxel_size, num_voxels_x, num_voxels_y, num_voxels_z,
                 x_min, y_min, z_min):
        """
        Creates a 3D voxel grid and updates the counters for each voxel when query points fall
        within them. Used to quantify visition of 3D space.
        ----------------------------------------------
        :param batch_size: int, size of batch
        :param device: compute device, e.g., 'cuda:0'
        :param voxel_size: float, length of voxel (assumed cube) dimension in meters
        :param num_voxels_x: number of voxels along x
        :param num_voxels_y: number of voxels along y
        :param num_voxels_z: number of voxels along z
        :param x_min: float, the minimum x position for the voxel grid
        :param y_min: float, the minimum y position for the voxel grid
        :param z_min: float, the minimum z position for the voxel grid
        """
        self.batch_size = batch_size
        self.device = device

        # Create 3D voxel counter
        self.voxel_size = voxel_size
        self.num_voxels_x = num_voxels_x
        self.num_voxels_y = num_voxels_y
        self.num_voxels_z = num_voxels_z
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min

        # (bxixjxk) Torch tensor of ints, where int value is the voxel visitation count
        # b = batch size
        # i = number of voxels along x
        # j = number of voxels along y
        # k = number of voxels along z
        self._voxel_counter = torch.zeros(batch_size, num_voxels_x, num_voxels_y, num_voxels_z,
                                          dtype=torch.int, device=device)
        # b dim Torch tensor of ints, where int value is the voxel visitation count for the current
        # query points
        self._voxel_count = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        # (bx3) Torch tensor of 3D center of corresponding voxel to query point
        self._voxel_centers = torch.zeros(batch_size, 3, device=device)

    @property
    def voxel_counter(self):
        """
        :return self._voxel_counter: (bxixjxk) Torch tensor of ints, 
                where int value is the voxel visitation count
                b = batch size
                i = number of voxels along x
                j = number of voxels along y
                k = number of voxels along z
        """
        return self._voxel_counter

    def zero_voxels(self, indices = None):
        """
        Zeros out the voxel visitation in the voxel grid along the specified indices.
        ------------------------------------------
        :param indices: 1D torch tensor of ints, the indices along the batch for which to 0
                        the entire voxel grid visition counts
        :param indices: None, zeros all voxel grid counts along the entire batch
        """
        if indices is None:
            self._voxel_counter *= 0
        else:
            self._voxel_counter[indices] *= 0

    def get_count(self, points):
        """
        Returns the voxel visitation count for the query points
        ------------------------------------------
        :param points: (bx3) Pytorch Tensor of floats, b number of 3D points
        :return self._voxel_count: (b) Pytorch Tensor of ints, where int is the times
                                   the query point has fallen within its associated voxel.
                                   Value can be 0 if point does not belong to any voxel
        :return self._voxel_centers: (bx3) 2D Torch tensor that contain the 3D voxel center
                                     of the voxel that associates with the query point
        """
        points_wp = wp.torch.from_torch(points, dtype=wp.vec3)
        voxel_counter_wp = wp.torch.from_torch(self._voxel_counter)

        # Zero out the voxel count and wrap with warp array before querying
        self._voxel_count *= 0
        voxel_count = wp.torch.from_torch(self._voxel_count)

        self._voxel_centers *= 0.
        voxel_centers = wp.torch.from_torch(self._voxel_centers)

        wp.launch(kernel=update_voxel_count,
                  dim=[self.batch_size, self.num_voxels_x, self.num_voxels_y, self.num_voxels_z],
                  inputs=[
                      self.voxel_size,
                      self.x_min,
                      self.y_min,
                      self.z_min,
                      points_wp,
                      voxel_counter_wp,
                      ],
                  outputs=[
                      voxel_count,
                      voxel_centers
                      ],
                  device=self.device)

        return (self._voxel_count, self._voxel_centers)

