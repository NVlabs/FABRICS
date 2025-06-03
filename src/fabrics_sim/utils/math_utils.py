import torch
import warp as wp

# Kernel for projecting batch_qdd such that all elements are within accel limits
@wp.kernel
def accel_constraint_proj_kernel(
        # inputs
        batch_qdd: wp.array(dtype=float, ndim=2),
        cspace_dim: int,
        accel_limits: wp.array(dtype=float, ndim=1),
        # output (can be the same as the corresponding batch_qdd input)
        batch_qdd_scaled: wp.array(dtype=float, ndim=2)):
    batch_index = wp.tid()
    min_scalar = float(1.)

@wp.kernel
def cholesky_factorization(
    # inputs
    A: wp.array(dtype=float, ndim=3),
    dim: int,
    # outputs
    L: wp.array(dtype=float, ndim=3)):

    batch_index, row, col = wp.tid()

    for i in range(dim):
        for j in range(i + 1):
            s = float(0.)
            for k in range(j):
                s = s + L[batch_index, i, k] * L[batch_index, j, k]
            if i == j:
                L[batch_index, i, j] = wp.sqrt(A[batch_index, i, i] - s)
            else:
                L[batch_index, i, j] = (1. / L[batch_index, j, j] * (A[batch_index, i, j] - s))


#    n = A.shape[0]
#    L = np.zeros_like(A, dtype=float)
#    
#    for i in range(n):
#        for j in range(i+1):
#            s = sum(L[i, k] * L[j, k] for k in range(j))
#            if i == j:
#                L[i, j] = np.sqrt(A[i, i] - s)
#            else:
#                L[i, j] = (1.0 / L[j, j] * (A[i, j] - s))
#    return L

@wp.kernel
def inverse_lower_triangular_matrix(
    # inputs
    L: wp.array(dtype=float, ndim=3),
    dim: int,
    # outputs
    L_inv: wp.array(dtype=float, ndim=3)):

    batch_index, row, col = wp.tid()

    for i in range(dim):
        L_inv[batch_index, i, i] = 1. / L[batch_index, i, i]
        for j in range(i + 1, dim):
            s = float(0.)
            for k in range(i, j):
                s = s - L[batch_index, j, k] * L_inv[batch_index, k, i]
            L_inv[batch_index, j, i] = s / L[batch_index, j, j]

# Custom pytorch operation that inverts a positive definite matrix
class InverseCholesky(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, A, A_inv, L, L_inv, device):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and ouputs
        ctx.A = wp.torch.from_torch(A)
        ctx.A_inv = wp.torch.from_torch(A_inv)
        ctx.L = wp.torch.from_torch(L)
        ctx.L_inv = wp.torch.from_torch(L_inv)
        ctx.device = device

        with ctx.tape:
            batch_size = A.shape[0]
            mat_dim = A.shape[1]
            wp.launch(kernel=cholesky_factorization,
                      dim=(batch_size, mat_dim, mat_dim),
                      inputs=[
                          ctx.A,
                          mat_dim
                          ],
                      outputs=[
                          ctx.L],
                      device=ctx.device)

            wp.launch(kernel=inverse_lower_triangular_matrix,
                  dim=(batch_size, mat_dim, mat_dim),
                  inputs=[
                      ctx.L,
                      mat_dim
                      ],
                  outputs=[
                      ctx.L_inv],
                  device=ctx.device)

        return wp.torch.to_torch(ctx.L_inv)

    @staticmethod
    def backward(ctx, adj_L_inv):
        ctx.L_inv.grad = wp.torch.from_torch(adj_L_inv)
    
        # Calculate gradients
        ctx.tape.backward()

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.L_inv]), None,
                None, None, None)


def inverse_pd_matrix(A, A_inv, L, L_inv, device):
    L_inv = InverseCholesky.apply(A, A_inv, L, L_inv, device)

    A_inv.copy_(torch.bmm(L_inv.transpose(1,2), L_inv))

