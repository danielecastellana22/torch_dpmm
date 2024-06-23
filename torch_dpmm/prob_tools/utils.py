import torch as th


__all__ = ['my_scatter_nd', 'log_normalise', 'batched_trace_square_mat', 'multidigamma', 'batch_outer_product']

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


def batch_outer_product(a, b):
    assert a.shape[-1] == b.shape[-1]
    return th.einsum('...aj,...bj->...ab', a.unsqueeze(-1), b.unsqueeze(-1))
