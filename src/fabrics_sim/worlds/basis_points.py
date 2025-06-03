import warp as wp
import torch
import numpy as np
import yaml
import time

from warp.sim.model import Mesh

@wp.kernel
def point_to_mesh_query(
    # inputs
    points: wp.array(dtype=wp.vec3, ndim=1),
    object_mesh: wp.array(dtype=wp.uint64, ndim=2),
    object_indicator: wp.array(dtype=wp.uint64, ndim=2),
    max_depth: float,
    # outputs
    signed_distance: wp.array(dtype=float, ndim=2),
    direction: wp.array(dtype=wp.vec3, ndim=2)):
    
    batch_index, point_index, mesh_index = wp.tid()

    if object_indicator[batch_index, mesh_index] == 0:
        return
    
    # Now query closest distance between point and mesh
    d = float(0.0)              # hit distance along ray
    u = float(0.0)              # hit  face barycentric u
    v = float(0.0)              # hit  face barycentric u
    sign = float(0.0)           # hit face sign, value > 0 if ray hit front of face
    n = wp.vec3()               # hit face normal
    f = int(0)                  # hit face index
    inside = float(0.)          # < 0 if inside the mesh, 0 >= otherwise
    bary_u = float(0.)
    bary_v = float(0.)

    # Query for distance between current body point and mesh
    got_dist = wp.mesh_query_point(object_mesh[batch_index, mesh_index],
                                   points[point_index], max_depth, inside, f, bary_u, bary_v)

    closest_point = wp.vec3()

    # If a positive return on distance check, then calculate direction
    # and signed distance
    if got_dist:
        # First get the closest point on the mesh to the body point
        closest_point = mesh_eval_position(object_mesh[batch_index, mesh_index],
                                           f, bary_u, bary_v)
    
        # Direction from sphere to closest point on the mesh
        n = wp.normalize(closest_point - points[point_index])

        # Create signed distance. Positive is outside or 0, negative is inside.
        d = wp.length(closest_point - points[point_index])
        d_signed = d * wp.sign(inside)

        # Save off distance if it is the smallest
        wp.atomic_min(signed_distance, batch_index, point_index, d_signed)

        # If saved smallest distance is the current one, then also save the
        # direction
        if abs(signed_distance[batch_index, point_index] - d_signed) < 1e-5:
            direction[batch_index, point_index] = n

class BasisPoints():
    def __init__(self, points_per_dim, coord_mins,
                 coord_maxs, object_ids, object_indicator,
                 device):
        """
        Creates a 3D grid of points as basis points from which closest distance
        and direction to mesh objects in the world is calculated
        ------------------------------------------
        :param points_per_dim: list or tuple of positive ints defining the number
                               of points to generate along each axis (x, y, z)
        :param coord_mins: list or tuple of floats defining the minimum position value
                           for each axis (x, y, z)
        :param coord_maxs: list or tuple of floats defining the maximum position value
                           for each axis (x, y, z)
        :param object_mesh_ids: 2D int Warp array referencing object meshes
        :param object_indicator: 2D Warp array of type uint64, indicating the presence
                                 of a Warp mesh in object_ids at corresponding index
                                 0=no mesh, 1=mesh
        :param device: str, defines device for tensors

        """
        assert (len(points_per_dim) == 3), "points_per_dim must be of length 3"
        assert (len(coord_mins) == 3), "coord_mins must be of length 3"
        assert (len(coord_maxs) == 3), "coord_max must be of length 3"
        assert (np.array(coord_maxs) > np.array(coord_mins)).min() == True,\
            "all maxs must be greater than mins"

        self.object_ids = object_ids
        self.object_indicator = object_indicator
        self.coord_mins = np.array(coord_mins)
        self.coord_maxs = np.array(coord_maxs)
        self.points_per_dim = np.array(points_per_dim)

        self.batch_size = object_ids.shape[0]

        self.device = device
        
        self.num_points = self.points_per_dim[0] *\
                          self.points_per_dim[1] *\
                          self.points_per_dim[2]

        self._points = None
        self.construct_point_grid()
    
        # Allocate warp distance and direction data
        self._signed_distance = wp.zeros(shape=(self.batch_size, self.num_points),
                                        dtype=float,
                                        device=self.device)
        self._direction = wp.zeros(shape=(self.batch_size, self.num_points),
                                  dtype=wp.vec3,
                                  device=self.device)

    def construct_point_grid(self):
        """
        Constructs the 3D grid of points as a Warp array

        """
        x = torch.linspace(self.coord_mins[0], self.coord_maxs[0],
                           self.points_per_dim[0], device=self.device)
        y = torch.linspace(self.coord_mins[1], self.coord_maxs[1],
                           self.points_per_dim[1], device=self.device)
        z = torch.linspace(self.coord_mins[2], self.coord_maxs[2],
                           self.points_per_dim[2], device=self.device)

        self.grid = torch.meshgrid(x, y, z, indexing='ij')
        grid_flat = []
        for i in range(3):
            grid_flat.append(self.grid[i].flatten())

        # Convert point data into a long tensor of 3D points
        points = torch.zeros(self.num_points, 3, device=self.device)

        for i in range(self.num_points):
            for j in range(3):
                points[i,j] = grid_flat[j][i]

        # Convert to Warp vec3 data type
        self._points = wp.torch.from_torch(points, dtype=wp.vec3)

    def query(self):
        """
        Performs the geometric query of closest distance and direction
        to mesh objects.
        """
        num_meshes = self.object_ids.shape[1]

        signed_distance_torch = 0. * wp.torch.to_torch(self._signed_distance) + 1e6
        self._signed_distance = wp.torch.from_torch(signed_distance_torch)

        direction_torch = 0. * wp.torch.to_torch(self._direction)
        self._direction = wp.torch.from_torch(direction_torch, dtype=wp.vec3)

        max_depth = 2.
        wp.launch(kernel=point_to_mesh_query,
                  dim=[self.batch_size, self.num_points, num_meshes],
                  inputs=[
                      self._points,
                      self.object_ids,
                      self.object_indicator,
                      max_depth,
                      ],
                  outputs=[
                      self._signed_distance,
                      self._direction
                      ],
                  device=self.device)

    def points(self):
        """
        Returns the 3D point grid as a Torch tensor
        ------------------------------------------
        :return points: n x 3 Torch tensor of n 3D points
        """
        return wp.torch.to_torch(self._points)

    def signed_distance(self):
        """
        Returns the signed distances as a Torch tensor
        ------------------------------------------
        :return signed_distance: b x n Torch tensor of distances where b is batch size
        """
        return wp.torch.to_torch(self._signed_distance)

    def direction(self):
        """
        Returns the direction to closest point as Torch tensor
        ------------------------------------------
        :return direction: b x n x 3 Torch tensor stating the 3D direction to cloest
                           point across batch and points
        """
        return wp.torch.to_torch(self._direction)

@wp.kernel
def point_to_point_cloud_query(
    # inputs
    basis_points: wp.array(dtype=wp.vec3, ndim=1),
    point_cloud: wp.array(dtype=wp.vec3, ndim=2),
    basis_points_transform: wp.array(dtype=wp.transform, ndim=1),
    max_depth: float,
    # outputs
    distance: wp.array(dtype=float, ndim=2),
    direction: wp.array(dtype=wp.vec3, ndim=2),
    basis_points_transformed: wp.array(dtype=wp.vec3, ndim=2)):
    
    batch_index, basis_point_index, point_cloud_point_index = wp.tid()

    # First transform basis point
    basis_point =\
        wp.transform_point(basis_points_transform[batch_index],
                           basis_points[basis_point_index])

    # Save to output array
    basis_points_transformed[batch_index, basis_point_index] = basis_point

    # Calculate distance between transformed basis point and point cloud point
    point2point_distance =\
        wp.length(point_cloud[batch_index, point_cloud_point_index] - basis_point)

    # Calculate direction between transformed basis point and point cloud point
    point2point_dir = \
        wp.normalize(point_cloud[batch_index, point_cloud_point_index] - basis_point)
    
    # Save off the distance if it is the current smallest
    wp.atomic_min(distance, batch_index, basis_point_index, point2point_distance)

    # Save off the direction if the associated distance was saved
    if abs(distance[batch_index, basis_point_index] - point2point_distance) < 1e-5:
        direction[batch_index, basis_point_index] = point2point_dir

class BasisPointsToPointCloud():
    def __init__(self, dim_size, points_per_dim, batch_size, device):
        
        self.dim_size = dim_size
        self.points_per_dim = points_per_dim
        self.batch_size = batch_size
        self.device = device
        
        self.num_basis_points = self.points_per_dim[0] *\
                                self.points_per_dim[1] *\
                                self.points_per_dim[2]

        self._basis_points = None
        self.construct_basis_points()
        
        # Allocate warp distance and direction data
        self._distance = wp.zeros(shape=(self.batch_size, self.num_basis_points),
                                        dtype=float,
                                        device=self.device)
        self._direction = wp.zeros(shape=(self.batch_size, self.num_basis_points),
                                  dtype=wp.vec3,
                                  device=self.device)

    def construct_basis_points(self):
        """
        Constructs the 3D grid of points as a Warp array

        """
        x = torch.linspace(-self.dim_size[0] / 2, self.dim_size[0] / 2,
                           self.points_per_dim[0], device=self.device)
        y = torch.linspace(-self.dim_size[1] / 2, self.dim_size[1] / 2,
                           self.points_per_dim[1], device=self.device)
        z = torch.linspace(-self.dim_size[2] / 2, self.dim_size[2] / 2,
                           self.points_per_dim[2], device=self.device)

        self.grid = torch.meshgrid(x, y, z, indexing='ij')
        grid_flat = []
        for i in range(3):
            grid_flat.append(self.grid[i].flatten())

        # Convert point data into a long tensor of 3D points
        points = torch.zeros(self.num_basis_points, 3, device=self.device)

        for i in range(self.num_basis_points):
            for j in range(3):
                points[i,j] = grid_flat[j][i]

        # Convert to Warp vec3 data type
        self._basis_points = wp.torch.from_torch(points, dtype=wp.vec3)
        self._basis_points_transformed =\
            wp.zeros(shape=(self.batch_size, self.num_basis_points),
                     dtype=wp.vec3,
                     device=self.device)

    def query_dist_to_point_cloud(self, basis_points_transform, point_cloud):
        # TODO: basis point transform is (x,y,z,rx,ry,rz,rw) convention
        
        distance_torch = 0. * wp.torch.to_torch(self._distance) + 1e6
        self._distance = wp.torch.from_torch(distance_torch)

        direction_torch = 0. * wp.torch.to_torch(self._direction)
        self._direction = wp.torch.from_torch(direction_torch, dtype=wp.vec3)

        num_point_cloud_points = point_cloud.shape[1]

        max_depth = 2.
        wp.launch(kernel=point_to_point_cloud_query,
                  dim=[self.batch_size, self.num_basis_points, num_point_cloud_points],
                  inputs=[
                      self._basis_points,
                      wp.torch.from_torch(point_cloud, dtype=wp.vec3),
                      # TODO: check torch to warp conversion here
                      # TODO: also check (xyzw) convention and order
                      wp.torch.from_torch(basis_points_transform, dtype=wp.transform),
                      max_depth,
                      ],
                  outputs=[
                      self._distance,
                      self._direction,
                      self._basis_points_transformed
                      ],
                  device=self.device)
                 
    def basis_points(self):
        """
        Returns the 3D point grid as a Torch tensor
        ------------------------------------------
        :return points: n x 3 Torch tensor of n 3D points
        """
        return wp.torch.to_torch(self._basis_points)
    
    def basis_points_transformed(self):
        """
        Returns the 3D point grid as a Torch tensor
        ------------------------------------------
        :return points: n x 3 Torch tensor of n 3D points
        """
        return wp.torch.to_torch(self._basis_points_transformed)

    def distance(self):
        """
        Returns the signed distances as a Torch tensor
        ------------------------------------------
        :return signed_distance: b x n Torch tensor of distances where b is batch size
        """
        return wp.torch.to_torch(self._distance)

    def direction(self):
        """
        Returns the direction to closest point as Torch tensor
        ------------------------------------------
        :return direction: b x n x 3 Torch tensor stating the 3D direction to cloest
                           point across batch and points
        """
        return wp.torch.to_torch(self._direction)






