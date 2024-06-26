import torch as th
from torch.distributions import Beta as th_Beta, MultivariateNormal
from torch.distributions import kl_divergence
from torch_dpmm.exp_family import Beta, FullNIW, DiagonalNIW, SingleNIW, SphericalNormal
from torch_dpmm.utils import  batched_trace_square_mat, multidigamma


B, K, D = 100, 20, 4


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


def wishart_kl_div(n0, V0, n1, V1, is_V1_diagonal=False, are_both_V_diagonals=False):
    D = V0.shape[-1]

    if are_both_V_diagonals:
        t1 = 0.5 * n1 * (-th.log(V1).sum(-1) + th.log(V0).sum(-1))
        t2 = 0.5 * n0 * ((V0 / V1).sum(-1) - D)
    elif is_V1_diagonal:
        t1 = 0.5 * n1 * (-th.log(V1).sum(-1) + th.linalg.slogdet(V0)[1])
        t2 = 0.5 * n0 * (batched_trace_square_mat(V0, diag_coeff=1 / V1) - D)
    else:
        V1_inv_V0 = th.linalg.solve(V1, V0)
        t1 = 0.5 * n1 * (-th.linalg.slogdet(V1)[1] + th.linalg.slogdet(V0)[1])
        t2 = 0.5 * n0 * (batched_trace_square_mat(V1_inv_V0) - D)

    t3 = th.special.multigammaln(0.5 * n1, D) - th.special.multigammaln(0.5 * n0, D)
    t4 = 0.5 * (n0 - n1) * multidigamma(0.5 * n0, D)

    return -t1 + t2 + t3 + t4


def mv_normal_wishart_kl_div(mu0, c0, n0, V0, mu1, c1, n1, V1, is_V0_diagonal=False, is_V1_diagonal=False):
    D = mu0.shape[-1]
    E_kl_mv = 0.5 * (c1 * n0 * batch_mahalanobis_dist(V0, mu1 - mu0, is_V0_diagonal) +
                     D * (c1 / c0 + th.log(c0) - th.log(c1) - 1))

    return E_kl_mv + wishart_kl_div(n0, V0, n1, V1, is_V1_diagonal=is_V1_diagonal,
                                    are_both_V_diagonals=is_V0_diagonal and is_V1_diagonal)


def test_beta_kl():
    # test beta_kl_div
    alpha_0, beta_0 = th.rand(K), th.rand(K)
    alpha_1, beta_1 = th.rand(K), th.rand(K)
    my_kl = Beta.kl_div(Beta.common_to_natural([alpha_0, beta_0]),
                        Beta.common_to_natural([alpha_1, beta_1]))
    th_kl = kl_divergence(th_Beta(alpha_0, beta_0), th_Beta(alpha_1, beta_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Beta_kl"


def test_unit_var_normal_kl():
    mu_0, lam_0 = th.randn(K, D), 1+th.rand(K)
    mu_1, lam_1 = th.randn(K, D), 1+th.rand(K)
    Sigma_0, Sigma_1 = th.diag_embed(1/lam_0.view(-1,1).expand(-1, D)), th.diag_embed(1/lam_1.view(-1,1).expand(-1, D))
    my_kl = SphericalNormal.kl_div(SphericalNormal.common_to_natural([mu_0, lam_0]),
                                   SphericalNormal.common_to_natural([mu_1, lam_1]))
    th_kl = kl_divergence(MultivariateNormal(mu_0, Sigma_0), MultivariateNormal(mu_1, Sigma_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Unit variance Normal kl"


def test_NIW_full_kl():
    a, b = th.randn(K, D, D) + 3 * th.diag_embed(th.ones(K, D)), th.randn(K, D, D) + 3 * th.diag_embed(th.ones(K, D))
    mu_0, lam_0, Phi_0, nu_0 = (th.rand(K, D), th.rand(K), th.einsum('bij,bkj->bik', a, a), D + th.randint(2, D, [K]))
    mu_1, lam_1, Phi_1, nu_1 = (th.rand(K, D), th.rand(K), th.einsum('bij,bkj->bik', b, b), D + th.randint(2, D, [K]))
    my_kl2 = mv_normal_wishart_kl_div(mu_0, lam_0, nu_0, th.linalg.inv(Phi_0), mu_1, lam_1, nu_1, th.linalg.inv(Phi_1),
                                      is_V1_diagonal=False, is_V0_diagonal=False)
    my_kl = FullNIW.kl_div(
        FullNIW.common_to_natural([mu_0, lam_0, Phi_0, nu_0]),
        FullNIW.common_to_natural([mu_1, lam_1, Phi_1, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with full cov matrices"


def test_NIW_diagonal_kl():
    mu_0, lam_0, Phi_0_diag, nu_0 = (th.rand(K, D), th.rand(K), th.rand(K, D), D + th.randint(2, D, [K]))
    mu_1, lam_1, Phi_1_diag, nu_1 = (th.rand(K, D), th.rand(K), th.rand(K, D), D + th.randint(2, D, [K]))
    my_kl2 = mv_normal_wishart_kl_div(mu_0, lam_0, nu_0, 1. / Phi_0_diag, mu_1, lam_1, nu_1, 1. / Phi_1_diag,
                                      is_V1_diagonal=True, is_V0_diagonal=True)
    my_kl = DiagonalNIW.kl_div(
        DiagonalNIW.common_to_natural([mu_0, lam_0, Phi_0_diag, nu_0]),
        DiagonalNIW.common_to_natural([mu_1, lam_1, Phi_1_diag, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with diagonal cov matrices"


def test_NIW_single_kl():
    mu_0, lam_0, p_0, nu_0 = (th.rand(K, D), th.rand(K), th.rand(K), D + th.randint(2, D, [K]))
    mu_1, lam_1, p_1, nu_1 = (th.rand(K, D), th.rand(K), th.rand(K), D + th.randint(2, D, [K]))

    Phi_0_diag, Phi_1_diag = p_0.view(-1,1) * th.ones(K, D), p_1.view(-1,1) * th.ones(K, D)
    my_kl2 = DiagonalNIW.kl_div(
        DiagonalNIW.common_to_natural([mu_0, lam_0, Phi_0_diag, nu_0]),
        DiagonalNIW.common_to_natural([mu_1, lam_1, Phi_1_diag, nu_1]))

    my_kl = SingleNIW.kl_div(
        SingleNIW.common_to_natural([mu_0, lam_0, p_0, nu_0]),
        SingleNIW.common_to_natural([mu_1, lam_1, p_1, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with diagonal cov matrices"


def test_o():
    mu_0, lam_0, Phi_0_diag, nu_0 = (th.rand(K, D), th.rand(K), th.rand(K, D), D + th.randint(2, D, [K]))
    mu_1, lam_1, Phi_1_diag, nu_1 = (th.rand(K, D), th.rand(K), th.rand(K, D), D + th.randint(2, D, [K]))
    Phi_0, Phi_1 = th.diag_embed(Phi_0_diag), th.diag_embed(Phi_1_diag)

    my_kl = DiagonalNIW.kl_div(
        DiagonalNIW.common_to_natural([mu_0, lam_0, Phi_0_diag, nu_0]),
        DiagonalNIW.common_to_natural([mu_1, lam_1, Phi_1_diag, nu_1]))

    my_kl2 = FullNIW.kl_div(
        FullNIW.common_to_natural([mu_0, lam_0, Phi_0, nu_0]),
        FullNIW.common_to_natural([mu_1, lam_1, Phi_1, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with diagonal cov matrices"

