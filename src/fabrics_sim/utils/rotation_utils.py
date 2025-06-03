import time

import torch
import warp as wp

wp.init()

@wp.kernel
def matrix_to_euler_kernel(
    # inputs
    matrix: wp.array(dtype=float, ndim=3),
    # outputs
    euler_angles: wp.array(dtype=float, ndim=2)):

    batch_index, angle_index = wp.tid()

    # Rotation about Z
    if angle_index == 0:
        euler_angles[batch_index, angle_index] =\
            wp.atan2(matrix[batch_index, 1, 0], matrix[batch_index, 0, 0])

    # Rotation about Y
    if angle_index == 1:
        euler_angles[batch_index, angle_index] = wp.asin(-matrix[batch_index, 2, 0])
        
    # Rotation about X
    if angle_index == 2:
        euler_angles[batch_index, angle_index] =\
            wp.atan2(matrix[batch_index, 2, 1], matrix[batch_index, 2, 2])

def matrix_to_euler(matrix):
    """
    Transforms rotation matrix to Euler ZYX
    """
    batch_size = matrix.shape[0]
    euler_angles = torch.zeros(batch_size, 3, device=matrix.device)

    wp.launch(kernel=matrix_to_euler_kernel,
              dim=(batch_size, 3),
              inputs=[wp.torch.from_torch(matrix)],
              outputs=[wp.torch.from_torch(euler_angles)],
              device=str(matrix.device))

    return euler_angles

@wp.kernel
def euler_to_matrix_kernel(
    # inputs
    euler_angles: wp.array(dtype=float, ndim=2),
    # outputs
    matrix: wp.array(dtype=float, ndim=3)):

    batch_index = wp.tid()

    # Rotation about Z
    psi = euler_angles[batch_index, 0]
    theta = euler_angles[batch_index, 1]
    phi = euler_angles[batch_index, 2]

    matrix[batch_index, 0, 0] = wp.cos(psi)*wp.cos(theta)
    matrix[batch_index, 0, 1] = wp.cos(psi)*wp.sin(theta)*wp.sin(phi) - wp.sin(psi)*wp.cos(phi)
    matrix[batch_index, 0, 2] = wp.cos(psi)*wp.sin(theta)*wp.cos(phi) + wp.sin(psi)*wp.sin(phi)
    matrix[batch_index, 1, 0] = wp.sin(psi)*wp.cos(theta)
    matrix[batch_index, 1, 1] = wp.sin(psi)*wp.sin(theta)*wp.sin(phi) + wp.cos(psi)*wp.cos(phi)
    matrix[batch_index, 1, 2] = wp.sin(psi)*wp.sin(theta)*wp.cos(phi) - wp.cos(psi)*wp.sin(phi)
    matrix[batch_index, 2, 0] = -wp.sin(theta)
    matrix[batch_index, 2, 1] = wp.cos(theta) * wp.sin(phi)
    matrix[batch_index, 2, 2] = wp.cos(theta) * wp.cos(phi)

def euler_to_matrix(euler_angles):
    batch_size = euler_angles.shape[0]
    matrix = torch.zeros(batch_size, 3, 3, device=euler_angles.device)

    wp.launch(kernel=euler_to_matrix_kernel,
              dim=batch_size,
              inputs=[wp.torch.from_torch(euler_angles)],
              outputs=[wp.torch.from_torch(matrix)],
              device=str(euler_angles.device))

    return matrix

@wp.kernel
def matrix_to_quaternion_kernel(
    # inputs
    matrix: wp.array(dtype=wp.float32, ndim=3),
    # outputs
    quaternion: wp.array(dtype=wp.float32, ndim=2)):

    batch_index = wp.tid()

    mat = wp.matrix(matrix[batch_index, 0, 0],
                    matrix[batch_index, 0, 1],
                    matrix[batch_index, 0, 2],
                    matrix[batch_index, 1, 0],
                    matrix[batch_index, 1, 1],
                    matrix[batch_index, 1, 2],
                    matrix[batch_index, 2, 0],
                    matrix[batch_index, 2, 1],
                    matrix[batch_index, 2, 2],
                    shape=(3,3), dtype=wp.float32)

    quat = wp.quat_from_matrix(mat)

    quaternion[batch_index, 0] = quat[3]
    for i in range(3):
        quaternion[batch_index, i + 1] = quat[i]

def matrix_to_quaternion(matrix):
    """
    Transforms rotation matrix to quaternion
    """
    batch_size = matrix.shape[0]
    quaternion = torch.zeros(batch_size, 4, device=matrix.device)

    wp.launch(kernel=matrix_to_quaternion_kernel,
              dim=batch_size,
              inputs=[wp.torch.from_torch(matrix)],
              outputs=[wp.torch.from_torch(quaternion)],
              device=str(matrix.device))

    return quaternion

@wp.kernel
def quaternion_to_matrix_kernel(
    # inputs
    quaternion: wp.array(dtype=wp.Float, ndim=2),
    # outputs
    matrix: wp.array(dtype=wp.Float, ndim=3)):

    batch_index = wp.tid()

    quat = wp.quaternion(quaternion[batch_index, 1],
                         quaternion[batch_index, 2],
                         quaternion[batch_index, 3],
                         quaternion[batch_index, 0],
                         dtype=wp.Float)

    mat = wp.quat_to_matrix(quat)

    for i  in range(3):
        for j in range(3):
            matrix[batch_index, i, j] = mat[i,j]

def quaternion_to_matrix(quaternion):

    batch_size = quaternion.shape[0]
    matrix = torch.zeros(batch_size, 3, 3, device=quaternion.device)

    wp.launch(kernel=quaternion_to_matrix_kernel,
              dim=batch_size,
              inputs=[wp.torch.from_torch(quaternion)],
              outputs=[wp.torch.from_torch(matrix)],
              device=str(quaternion.device))
    
    return matrix

if __name__ == "__main__":

    from pytorch3d import transforms
    batch_size = 1000
    device = 'cuda'

    # Testing matrix to Euler ZYX------------------------------------------------------------------
    x = torch.nn.functional.normalize(2. * (torch.rand(batch_size, 3, device=device) - 0.5), dim=1)
    y1 = torch.nn.functional.normalize(2. * (torch.rand(batch_size, 3, device=device) - 0.5), dim=1)
    y = torch.nn.functional.normalize(torch.cross(x, y1, dim=1), dim=1)
    z = torch.cross(x, y, dim=1)

    rotation_matrix = torch.zeros(batch_size, 3, 3, device=device)
    rotation_matrix[:, 0] = x
    rotation_matrix[:, 1] = y
    rotation_matrix[:, 2] = z

    start = time.time()
    euler_angles = matrix_to_euler(rotation_matrix)
    print('custom euler angles', euler_angles)
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    euler_angles_gt = transforms.matrix_to_euler_angles(rotation_matrix, "ZYX")
    print('euler angles gt', euler_angles_gt)
    torch.cuda.synchronize()
    print(time.time() - start)

    # Testing matrix to quaternion------------------------------------------------------------------
    start = time.time()
    quaternion = matrix_to_quaternion(rotation_matrix)
    print('custom quat', quaternion, quaternion.norm())
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    quat_gt = transforms.matrix_to_quaternion(rotation_matrix)
    print('quat gt', quat_gt, quat_gt.norm())
    torch.cuda.synchronize()
    print(time.time() - start)

    # Testing Euler to matrix
    start = time.time()
    matrix = euler_to_matrix(euler_angles)
    print('custom matrix', matrix)
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    matrix_gt = transforms.euler_angles_to_matrix(euler_angles, "ZYX")
    print('matrix gt', matrix_gt)
    torch.cuda.synchronize()
    print(time.time() - start)

    # Testing quaternion to matrix
    start = time.time()
    matrix = quaternion_to_matrix(quaternion)
    print('custom matrix', matrix)
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    matrix_gt = transforms.quaternion_to_matrix(quaternion)
    print('matrix gt', matrix_gt)
    torch.cuda.synchronize()
    print(time.time() - start)

