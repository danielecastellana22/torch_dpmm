import torch as th
from typing import Optional, Union


__all__ = ['my_scatter_nd', 'log_normalise', 'batched_trace_square_mat', 'multidigamma',
           'batch_mahalanobis_dist', 'batch_outer_product', 'batch_cholesky_update']

def my_scatter_nd(data, idx_tensor, shape):
    return th.sparse_coo_tensor(idx_tensor, data, shape).coalesce().to_dense().to(data.device)


# we assume that we normalise over the last axis
def normalise(tab: th.Tensor):
    return tab / th.sum(tab,-1,keepdim=True)


# we assume that we normalise over the last axis
def log_normalise(log_tab):
    log_max = th.max(log_tab, -1, keepdim=True).values
    scaled_tab = log_tab-log_max
    log_Z = th.log(th.sum(th.exp(scaled_tab), -1, keepdim=True)) + log_max
    return (log_tab - log_Z), log_Z


def batched_trace_square_mat(mat, diag_coeff=None):
    if diag_coeff is not None:
        to_sum = th.diagonal(mat, dim1=-2, dim2=-1) * diag_coeff
    else:
        to_sum = th.diagonal(mat, dim1=-2, dim2=-1)
    return to_sum.sum(-1)


def multidigamma(input: th.Tensor, p):
    d_shape = [1]*input.ndim
    dvec = th.arange(1, p + 1, dtype=th.float, device=input.device).view(d_shape+[p])
    aux = input.unsqueeze(-1) + 0.5 * (1 - dvec)
    return th.special.digamma(aux).sum(-1)


def batch_mahalanobis_dist(M, v, is_M_diagonal):
    r"""
        Compute v^T M^ v. We assume that M and v have been already broadcasted.
    """
    if is_M_diagonal:
        assert M.ndim == v.ndim
        return th.sum((v ** 2) * M, -1)
    else:
        assert M.ndim == v.ndim+1
        M = M.expand(v.shape[:-1] + (-1,) * (M.ndim - v.ndim + 1))
        return (v.unsqueeze(-2) @ (M @ v.unsqueeze(-1))).squeeze(-1, -2)


def batch_outer_product(a, b):
    assert a.shape[-1] == b.shape[-1]
    return th.einsum('...aj,...bj->...ab', a.unsqueeze(-1), b.unsqueeze(-1))


def batch_cholesky_update(L: th.Tensor, x: th.Tensor,  beta: Optional[th.Tensor | float] = 1.0) -> th.Tensor:
    '''
    This function computes the Cholesky factor of (LL^T + beta*xx^T). It supports batch updates.
    The code is mostly taken from https://brentyi.github.io/fannypack/utils/potpourri/#fannypack.utils.cholupdate
    Args:
        L: tensors of size (*, D, D) which contains batched Cholesky facotrs.
        beta: tensors of size * which contains the weights for the updates.
        x: tensors of size (*, D)

    Returns:
        (th.Tensor): The Cholesky factor of (LL^T + beta*xx^T).
    '''
    # Expected shapes: (*, dim, dim) and (*, dim)
    batch_dims = L.shape[:-2]
    matrix_dim = x.shape[-1]
    assert x.shape[:-1] == batch_dims
    assert matrix_dim == L.shape[-1] == L.shape[-2]

    # Flatten batch dimensions, and clone for tensors we need to mutate
    L = L.reshape((-1, matrix_dim, matrix_dim))
    x = x.reshape((-1, matrix_dim)).clone()
    L_out_cols = []

    sign_out: Union[float, th.Tensor]

    if isinstance(beta, float):
        beta = th.tensor(beta, device=x.device)

    x = x * th.sqrt(th.abs(beta))
    sign_out = th.sign(beta)

    # Cholesky update; mostly copied from Wikipedia:
    # https://en.wikipedia.org/wiki/Cholesky_decomposition
    for k in range(matrix_dim):
        r = th.sqrt(L[:, k, k] ** 2 + sign_out * x[:, k] ** 2)
        c = (r / L[:, k, k])[:, None]
        s = (x[:, k] / L[:, k, k])[:, None]

        # We build output column-by-column to avoid in-place modification errors
        L_out_col = th.zeros_like(L[:, :, k])
        L_out_col[:, k] = r
        L_out_col[:, k + 1:] = (L[:, k + 1:, k] + sign_out * s * x[:, k + 1:]) / c
        L_out_cols.append(L_out_col)

        # We clone x at each iteration, also to avoid in-place modification errors
        x_next = x.clone()
        x_next[:, k + 1:] = c * x[:, k + 1:] - s * L_out_col[:, k + 1:]
        x = x_next

    # Stack columns together
    L_out = th.stack(L_out_cols, dim=2)

    # Unflatten batch dimensions and return
    return L_out.reshape(batch_dims + (matrix_dim, matrix_dim))