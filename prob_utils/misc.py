import torch as th

__all__ = ['my_scatter_nd', 'log_normalise', 'batched_trace_square_mat', 'multidigamma',
           'batch_mahalanobis_dist', 'batch_outer_product']

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


def multidigamma(input:th.Tensor, p):
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


if __name__ == '__main__':
    BS, K, D = 100, 20, 4
    # test multidigamma
    from torch.distributions.wishart import _mvdigamma
    for p in range(2, 12):
        v = p + 10*th.rand(BS, D)
        my_val = multidigamma(v, p)
        torch_val = _mvdigamma(v, p)
        assert th.all(th.isclose(my_val, torch_val)), "mutligamma fails"

    # test batch mahalanobis
    from torch.distributions.multivariate_normal import _batch_mahalanobis

    # diagonal case
    M = 1 + 5 * th.rand(K, D)
    v = th.randn(BS, K, D)
    my_val = batch_mahalanobis_dist(M.unsqueeze(0), v, is_M_diagonal=True)
    torch_val = _batch_mahalanobis(th.diag_embed(1/M.sqrt()), v)
    assert th.all(th.isclose(my_val, torch_val)), "diagonal batch_mahalanobis fails"

    # full case (stable)
    a =  th.rand(K, D, D)
    M = th.einsum('...ij,...kj->...ik', a, a) + 5 * th.diag_embed(th.ones(K, D))
    l = th.linalg.cholesky(M)
    M_inv = th.linalg.inv(M)
    v = th.randn(BS, K, D)
    my_val = batch_mahalanobis_dist(M_inv.unsqueeze(0), v, is_M_diagonal=False)
    torch_val = _batch_mahalanobis(l, v)
    assert th.all(th.isclose(my_val, torch_val)), "full batch_mahalanobis fails"

    # full case (stable) multiple batch dimensin
    a = th.rand(K, K, D, D)
    M = th.einsum('...ij,...kj->...ik', a, a) + 5 * th.diag_embed(th.ones(K, K, D))
    l = th.linalg.cholesky(M)
    M_inv = th.linalg.inv(M)
    v = th.randn(BS, K, K, D)
    my_val = batch_mahalanobis_dist(M_inv.unsqueeze(0), v, is_M_diagonal=False)
    torch_val = _batch_mahalanobis(l, v)
    assert th.all(th.isclose(my_val, torch_val)), "full batch_mahalanobis fails"

    # TODO: test batch_outer_product
    # TODO: test batched_trace_sqaure_mat
