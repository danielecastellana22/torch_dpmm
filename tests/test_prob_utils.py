import torch as th
from torch_dpmm.utils import *

BS, K, D = 100, 20, 4


def test_multivariate_digamma():
    # test multidigamma
    from torch.distributions.wishart import _mvdigamma
    for p in range(2, 12):
        v = p + 10*th.rand(BS, D)
        my_val = multidigamma(v, p)
        torch_val = _mvdigamma(v, p)
        assert th.all(th.isclose(my_val, torch_val)), "mutligamma fails"


def test_batch_mahalanobis():
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
