import numpy as np
import time
import torch
import warp as wp
from fabrics_sim.utils.math_utils import cholesky_factorization, inverse_lower_triangular_matrix, inverse_pd_matrix

wp.init()

batch_size = 1
mat_dim = 10
device = 'cuda'

rand_mat = torch.rand(batch_size, mat_dim, mat_dim, device=device)
A = torch.bmm(rand_mat, rand_mat.transpose(1, 2))

#eigenvalues = torch.linalg.eigvals(A)[0]

L = torch.zeros_like(A)
            
wp.launch(kernel=cholesky_factorization,
          dim=(batch_size, mat_dim, mat_dim),
          inputs=[
              wp.torch.from_torch(A),
              mat_dim
              ],
          outputs=[
              wp.torch.from_torch(L)],
          device=device)

print(L)

L_check = np.linalg.cholesky(A.detach().cpu().numpy()[0])
print('L_check', L_check)

# Inverse L----------------------------------
L_inv = torch.zeros_like(L)

wp.launch(kernel=inverse_lower_triangular_matrix,
          dim=(batch_size, mat_dim, mat_dim),
          inputs=[
              wp.torch.from_torch(L),
              mat_dim
              ],
          outputs=[
              wp.torch.from_torch(L_inv)],
          device=device)

print(L_inv)

print('L inv check', torch.inverse(L))

#A_inv = torch.zeros_like(A)
L = torch.zeros_like(A)
L_inv = torch.zeros_like(A)


# capture
g = torch.cuda.CUDAGraph()
torch_stream = torch.cuda.Stream(device=device)

# make warp use the same stream
warp_stream = wp.stream_from_torch(torch_stream)

with wp.ScopedStream(warp_stream), torch.cuda.graph(g, stream=torch_stream):
    A_inv = inverse_pd_matrix(A, A_inv, L, L_inv, device)

print('A inv', A_inv)

A_inv_check = torch.inverse(A)

print('A inv check', A_inv_check)

g.replay()

start = time.time()
for i in range(100):
    g.replay()
torch.cuda.synchronize()
custom_time = (time.time() - start) / 100.


start = time.time()
for i in range(100):
    start = time.time()
    A_inv_check = torch.inverse(A)
torch.cuda.synchronize()
torch_time = (time.time() - start) / 100.

print('custom pytorch op inverse time', custom_time)
print('native torch inverse time', torch_time)
input('paused')

# Runs through a few cycles checking results against Torch's inverse
for i in range(100):
    rand_mat = torch.rand(batch_size, mat_dim, mat_dim, device=device)
    A.copy_(torch.bmm(rand_mat, rand_mat.transpose(1, 2))) # maintains A's address
    g.replay()
    print('A custom inv', A_inv)
    A_inv_check = torch.inverse(A)
    print('A inv check', A_inv_check)
    input('p')

