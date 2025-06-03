import warp as wp
import torch
import numpy as np
import yaml
import time

from warp.sim.model import Mesh

from fabrics_sim.utils.path_utils import get_world_path, get_object_urdf_path

@wp.kernel
def transform_mesh_points(
    # inputs
    num_meshes: int,
    mesh_points: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=int),
    mesh_object_transforms: wp.array(dtype=wp.mat44),
    mesh_object_starting_face_index: wp.array(dtype=int),
    mesh_object_ending_face_index: wp.array(dtype=int),
    # outputs
    transformed_mesh_points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    for i in range(num_meshes):
        if tid >= mesh_object_starting_face_index[i] and tid <= mesh_object_ending_face_index[i]:
            transformed_mesh_points[tid] =\
                    wp.transform_point(mesh_object_transforms[i], mesh_points[tid])
            #break

class WorldMeshModel():
    def __init__(self, objects_name, objects_face_indices, objects_vertices):
        """
        Constructor. Takes in objects and creates a single Warp Mesh object
        with kernels that can update this single Warp Mesh object given pose
        updates. 
        ------------------------------------------
        @param objects: list of trimesh objects
        """

        for i in range(0, len(objects_name)):
            num_vertices = np.array(objects_face_indices[i]).max() + 1
            if i == 0:
                self.object_starting_face_index = [0]
                self.object_ending_face_index = [num_vertices - 1]
            else:
                self.object_starting_face_index.append(
                        self.object_ending_face_index[-1] + 1)
                self.object_ending_face_index.append(self.object_starting_face_index[-1] +\
                        num_vertices - 1)

        # Stack all face indices together into one long list, shifting them accordingly
        # for every object.
        for i in range(len(self.object_starting_face_index)):
            objects_face_indices[i] += self.object_starting_face_index[i]

        # Stack into a single 2D array and flatten.
        objects_face_indices_stacked = np.concatenate(objects_face_indices, axis=0).flatten()

        # Stack all object vertices into single 2D array
        objects_vertices_stacked = np.concatenate(objects_vertices, axis=0)
        objects_vertices_stacked = list(objects_vertices_stacked)

        world_mesh = Mesh(objects_vertices_stacked, objects_face_indices_stacked)
        world_mesh.finalize(device='cuda')
        self.world_mesh = world_mesh.mesh
        self.world_mesh.refit() # run this so it can be used in raycasting.
       
        # Allocate reference points which we will transform from.
        self.world_mesh_points = wp.clone(self.world_mesh.points)

        # Convert starting and ending index lists to warp arrays
        self.object_starting_face_index =\
                wp.array(self.object_starting_face_index, dtype=int, device='cuda')
        self.object_ending_face_index =\
                wp.array(self.object_ending_face_index, dtype=int, device='cuda')

    def update_mesh(self, objects_transforms_list, robot_pose):
        # First find the transfrom from robot to objects, T_r_o
        for i in range(len(objects_transforms_list)):
            # Robot world expressed in robot
            T_r_w = np.linalg.inv(robot_pose)
            # Object expressed in robot
            T_w_o = objects_transforms_list[i]
            T_r_o = np.dot(T_r_w, T_w_o)
            objects_transforms_list[i] = T_r_o
       
        # Convert list of object poses expressed in robot into warp array.
        objects_transforms = wp.array(objects_transforms_list, dtype=wp.mat44, device='cuda')

        wp.launch(kernel=transform_mesh_points,
                  dim=self.world_mesh_points.shape[0],
                  inputs=[
                      len(objects_transforms_list),
                      self.world_mesh_points,
                      self.world_mesh.indices,
                      objects_transforms,
                      self.object_starting_face_index,
                      self.object_ending_face_index
                      ],
                  outputs=[
                      self.world_mesh.points
                      ],
                  device='cuda')

        # Run a refit so we can do raycasting.
        self.world_mesh.refit()

    @property
    def mesh(self):
        return self.world_mesh

    @property
    def mesh_points(self):
        return self.world_mesh_points

@wp.kernel
def transform_single_mesh_points(
    # inputs
    robot_body_points: wp.array(dtype=wp.vec3),
    transform: wp.transform,
    scaling: wp.vec3,
    # outputs
    transformed_robot_body_points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    # First scale the body points expressed in body-centric coordinate system
    transformed_robot_body_points[tid] = cw_mul(scaling, robot_body_points[tid])

    # Now transform the scaled points
    transformed_robot_body_points[tid] =\
        wp.transform_point(transform, transformed_robot_body_points[tid])

class WorldMeshesModel():
    def __init__(self, batch_size, max_objects_per_env, device,
                 world_filename=None, world_dict=None):
        """
        Constructor for world object which includes any number of objects, their meshes,
        and pose locations. This data can be used for collision avoidance functions leveraging
        Warp tooling.
        ------------------------------------------
        :param batch_size: batch size of world
        :param max_objects_per_env: int, number of allowed mesh objects per env in batch
        :param device: str, targeted device, e.g., 'cuda:0'
        :param world_filename: str, name of the world file
        :param world_dict: dictionary of world objects
        """
        self.original_objects = dict()
        self.objects = dict()
        self.device = device
        self.batch_size = batch_size
        self.mesh_ids = torch.zeros(batch_size, max_objects_per_env, dtype=torch.int64,
                                    device=device)
        self.mesh_indicator = torch.zeros(batch_size, max_objects_per_env, dtype=torch.int64,
                                    device=device)

        #assert(not (world_filename == None and world_dict == None)),\
        #    "Both world_filename and world_dict cannot be None"
        
        if world_filename is not None:
            self.load_world_from_file(world_filename)
        else:
            self.load_world(world_dict)

    def load_world_from_file(self, world_filename):
        """
        Constructs a world object which includes any number of objects, their meshes,
        and pose locations. This data can be used for collision avoidance functions leveraging
        Warp tooling.
        ------------------------------------------
        :param world_filename: str, name of the world file
        """
        world_path = get_world_path(world_filename)
        with open(world_path, 'r') as file:
            world_dict = yaml.safe_load(file)

        self.load_world(world_dict)

    def load_world(self, world_dict):
        """
        Constructs a world object which includes any number of objects, their meshes,
        and pose locations. This data can be used for collision avoidance functions leveraging
        Warp tooling.
        ------------------------------------------
        :param world_dict: dict, containing information about world
        """

        # Early out if there is no world dict (empty world)
        if world_dict is None:
            return
       
        for obj_name, obj_data in world_dict.items():
            # If object was not loaded from file before, called 'type', then create a new
            # allocation for this object
            start = time.time()
            if obj_data['type'] not in self.original_objects:
                new_object_model = self.create_object_model(obj_data['type'])
                self.original_objects[obj_data['type']] = new_object_model

            object_model = self.original_objects[obj_data['type']]
            # Create new mesh model that we will use to transform and refit.
            # NOTE: we need to copy the mesh points and create the Mesh from those such
            # that we can retain the original mesh points expressed in body-fixed coordinates
            # and calculate new point positions based on transforms.
            object_model_points = wp.zeros_like(object_model.shape_geo_src[0].mesh.points)
            wp.copy(object_model_points, object_model.shape_geo_src[0].mesh.points)
            object_mesh = wp.Mesh(object_model_points,
                                  object_model.shape_geo_src[0].mesh.indices)

            # Pull out the object transform
            x_form = [float(x) for x in obj_data['transform'].split()]
            object_transform = wp.transform(p=(x_form[0], x_form[1], x_form[2]),
                q=(x_form[3], x_form[4], x_form[5], x_form[6]))
            
            # Pull out the object scaling
            scaling = [float(x) for x in obj_data['scaling'].split()]
            object_scaling = wp.vec3(scaling[0], scaling[1], scaling[2])

            # Transform points.
            wp.launch(kernel=transform_single_mesh_points,
                      dim=len(object_mesh.points),
                      inputs=[
                          object_model.shape_geo_src[0].mesh.points,
                          object_transform,
                          object_scaling],
                      outputs=[object_mesh.points],
                      device=self.device)

            # Refit the object with its transformed points
            # TODO: if object poses are changing, then need to call refit as well. Need
            # to write functionality around moving objects around.
            object_mesh.refit()
            
            # Add object model, object transform, and object_mesh in dictionary.
            self.add_object(obj_name, object_model, object_transform, object_mesh, object_scaling,
                            obj_data['env_index'])
    
    def create_object_model(self, object_name):
        """
        Loads the object model urdf.
        -----------------------------
        :param object_name: name of object (should be same as folder name and urdf name)
        :return model: Warp model
        """

        # Load the object.
        builder = wp.sim.ModelBuilder()

        object_urdf_filename = get_object_urdf_path(object_name)
        initial_rotation = wp.quat(0., 0., 0., 1.)
        initial_position = wp.vec3(0., 0., 0.)
        initial_transform = wp.transform(initial_position, initial_rotation)

        print('importing object')
        wp.sim.parse_urdf(object_urdf_filename, builder, initial_transform)

        # Finalize model.
        print('finalizing model')
        model = builder.finalize(device=self.device)

        return model

    def add_object(self, object_name, object_model, object_transform, object_mesh, object_scaling,
                   env_index):
        """
        Adds an object and its relevant data to a dictionary of objects in the world.
        ------------------------------------------
        :param object_name: str, unique name for the object
        :param object_model: Warp model of the object
        :param object_transform: Warp transform (x,y,z,rx,ry,rz,w)
        :param object_mesh: Warp mesh object
        :param object_scaling: Warp vec3 object that is a scaling vector (sx,sy,sz) on the
                               object mesh
        :param env_index: int, indicates which env to associate mesh with indicated by env index
        """
        if object_name in self.objects:
            raise ValueError('Object already exists!')

        # Enter object data into object dictionary
        self.objects[object_name] =\
            { 'model': object_model,
              'transform': object_transform,
              'object_mesh': object_mesh,
              'object_scaling': object_scaling,
              'env_index': env_index}

        # If object is being assigned to all envs indicated by env_index == "all", then cycle
        # through and add it to all envs
        if env_index == 'all':
            for i in range(self.batch_size):
                # Find the first False entry in mesh indicator along env index slice to see where
                # to insert the new mesh ID
                new_mesh_index = (self.mesh_indicator[i] == 0).nonzero(as_tuple=True)[0][0]
                self.mesh_ids[i, new_mesh_index] = object_mesh.id
                self.mesh_indicator[i, new_mesh_index] = 1
        # Otherwise, just find the first 0 in mesh indicator along the env_index and insert
        # the new mesh id and update indicator to 1
        else:
            # Find the first False entry in mesh indicator along env index slice to see where
            # to insert the new mesh ID
            new_mesh_index = (self.mesh_indicator[env_index] == 0).nonzero(as_tuple=True)[0][0]
            self.mesh_ids[env_index, new_mesh_index] = object_mesh.id
            self.mesh_indicator[env_index, new_mesh_index] = 1

    def get_object_ids(self):
        """
        Returns the mesh reference ids for the collection of objects in the world.
        ------------------------------------------
        :return object_ids: 2D Warp array of type uint64, containing Warp mesh references
        :return object_indicator: 2D Warp array of type uint64, indicating the presence
                                  of a Warp mesh in object_ids at corresponding index
                                  0=no mesh, 1=mesh
        """
        object_ids = wp.from_torch(self.mesh_ids, dtype=wp.uint64)
        object_indicator = wp.from_torch(self.mesh_indicator, dtype=wp.uint64)

        return (object_ids, object_indicator)

    def get_object_scaling(self, object_name):
        """
        Returns the object mesh scaling vector.
        ------------------------------------------
        :return object_scaling: 1D Pytorch tensor of object scaling vector (sx,sy,sz)
        """
        return torch.as_tensor(self.objects[object_name]['object_scaling'], device=self.device)

    def get_object_transform(self, object_name):
        """
        Returns the object transform.
        ------------------------------------------
        :return object_transform: 1D Pytorch tensor of object transform vector (x,y,z,rx,ry,rz,w).
        """
        return torch.as_tensor(self.objects[object_name]['transform'], device=self.device)

    def get_object_names(self):
        """
        Returns the list of object names in the world.
        ------------------------------------------
        :return object_names: list of str, object names in the world.
        """
        object_names = []
        for object_name, object_data in self.objects.items():
            object_names.append(object_name)
        return object_names


