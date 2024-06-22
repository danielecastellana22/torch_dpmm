import torch as th
from .bnpy_test import *


def create_data_and_bnpy_model(is_diagonal):
    BS, K, D = 100, 20, 4

    # data
    x = th.rand(BS, D)

    # priors values
    alphaDP = th.tensor(4).float()
    tau0 = th.randn(D).float()
    c0 = 2 + th.rand(size=[1])
    n0 = D + th.randint(D, size=[1]).float()
    # B0 is always diagonal
    B0 = th.randn(D) ** 2
    B0_bnpy = th.diag_embed(B0)

    # var params of the stick
    u = th.rand(K)
    v = th.rand(K)

    # var params of mu
    tau = th.randn(K, D)
    c = 2 + th.rand(K)

    # var params of Lam
    n = D + th.randint(D, size=[K]).float()

    if is_diagonal:
        # diagonal with random var params
        B = 5 + th.randn(K, D) ** 2
        B_bnpy = th.diag_embed(B)
    else:
        b = th.randn(K, D, D)
        B = 5 * th.diag_embed(th.ones(K, D)) + th.einsum('bij,bkj->bik', b, b)
        B_bnpy = B

    bnpy_hmodel = get_bnpy_impl(K, D, u, v, alphaDP, tau, c, n, B_bnpy, tau0, c0, n0, B0_bnpy)
    return bnpy_hmodel, x, (u, v), (tau, c, B, n,), (alphaDP,), (tau0, c0, B0, n0)