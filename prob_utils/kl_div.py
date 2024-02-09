import torch as th
from torch.linalg import solve, slogdet
import warnings
from .misc import multidigamma, batched_trace_square_mat, batch_mahalanobis_dist

# we assume correct shapes and valid values (i.e. they satisfy the constraints required by the distribution)

__all__ = ['beta_kl_div', 'gamma_kl_div', 'mv_normal_kl_div', 'wishart_kl_div', 'mv_normal_wishart_kl_div']

def beta_kl_div(alpha0, beta0, alpha1, beta1):
    sum_params_p = alpha0 + beta0
    sum_params_q = alpha1 + beta1
    t1 = alpha1.lgamma() + beta1.lgamma() + (sum_params_p).lgamma()
    t2 = alpha0.lgamma() + beta0.lgamma() + (sum_params_q).lgamma()
    t3 = (alpha0 - alpha1) * th.digamma(alpha0)
    t4 = (beta0 - beta1) * th.digamma(beta0)
    t5 = (sum_params_q - sum_params_p) * th.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5


def gamma_kl_div(alpha0, beta0, alpha1, beta1):
    t1 = alpha1 * (beta0 / beta1).log()
    t2 = th.lgamma(alpha1) - th.lgamma(alpha0)
    t3 = (alpha0 - alpha1) * th.digamma(alpha0)
    t4 = (beta1 - beta0) * (alpha0 / beta0)
    return t1 + t2 + t3 + t4


def mv_normal_kl_div(mu0, Sigma0, mu1, Sigma1, is_Sigma1_diagonal=False, are_both_Sigma_diagonals=False):
    D = mu0.shape[-1]
    mu_dif = mu1 - mu0

    if are_both_Sigma_diagonals:
        assert (Sigma0.shape[-1] == Sigma1.shape[-1])
        t1 = (Sigma0 / Sigma1).sum(-1)
        t2 = (mu_dif ** 2 / Sigma1).sum(-1)
        t3 = th.log(Sigma1).sum(-1) - th.log(Sigma0).sum(-1)
    elif is_Sigma1_diagonal:
        t1 = batched_trace_square_mat(Sigma0, diag_coeff=(1 / Sigma1))
        t2 = (mu_dif.unsqueeze(-2) @ (mu_dif / Sigma1).unsqueeze(-1)).squeeze(-1, -2)
        t3 = th.log(Sigma1).sum(-1) - slogdet(Sigma0)[1]
    else:
        warnings.warn('Current implementation of KL between Guassians with full covariance matrix can be unstable!')
        t1 = batched_trace_square_mat(solve(Sigma1, Sigma0))
        t2 = (mu_dif.unsqueeze(-2) @ (solve(Sigma1, mu_dif)).unsqueeze(-1)).squeeze(-1, -2)
        t3 = slogdet(Sigma1)[1] - slogdet(Sigma0)[1]

    return 0.5 * (t1 + t2 + t3 - D)


def wishart_kl_div(n0, V0, n1, V1, is_V1_diagonal=False, are_both_V_diagonals=False):
    D = V0.shape[-1]

    if are_both_V_diagonals:
        t1 = 0.5 * n1 * (-th.log(V1).sum(-1) + th.log(V0).sum(-1))
        t2 = 0.5 * n0 * ((V0 / V1).sum(-1) - D)
    elif is_V1_diagonal:
        t1 = 0.5 * n1 * (-th.log(V1).sum(-1) + slogdet(V0)[1])
        t2 = 0.5 * n0 * (batched_trace_square_mat(V0, diag_coeff=1 / V1) - D)
    else:
        warnings.warn('Current implementation of KL between Wishart with full covariance matrix can be unstable!')
        V1_inv_V0 = solve(V1, V0)
        t1 = 0.5 * n1 * (-slogdet(V1)[1] + slogdet(V0)[1])
        t2 = 0.5 * n0 * (batched_trace_square_mat(V1_inv_V0) - D)

    t3 = th.special.multigammaln(0.5 * n1, D) - th.special.multigammaln(0.5 * n0, D)
    t4 = 0.5 * (n0 - n1) * multidigamma(0.5 * n0, D)

    return -t1 + t2 + t3 + t4


def mv_normal_wishart_kl_div(mu0, c0, n0, V0, mu1, c1, n1, V1, is_V0_diagonal=False, is_V1_diagonal=False):
    D = mu0.shape[-1]
    E_kl_mv = 0.5 * (c1 * n0 * batch_mahalanobis_dist(V0, mu1 - mu0, is_V0_diagonal) +
                     D*(c1/c0 + th.log(c0) - th.log(c1) - 1))

    return E_kl_mv + wishart_kl_div(n0, V0, n1, V1, is_V1_diagonal=is_V1_diagonal,
                                    are_both_V_diagonals=is_V0_diagonal and is_V1_diagonal)


if __name__ == '__main__':
    from torch.distributions import *

    B, K, D = 100, 20, 4

    ####################################################################################################################
    # test beta_kl_div
    alpha_0, beta_0 = th.rand(K), th.rand(K)
    alpha_1, beta_1 = th.rand(K), th.rand(K)
    my_kl = beta_kl_div(alpha_0, beta_0, alpha_1, beta_1)
    th_kl = kl_divergence(Beta(alpha_0, beta_0), Beta(alpha_1, beta_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()),  "Beta_kl"
    ####################################################################################################################

    ####################################################################################################################
    # test gamma_kl_div
    alpha_0, beta_0 = th.rand(K), th.rand(K)
    alpha_1, beta_1 = th.rand(K), th.rand(K)
    my_kl = gamma_kl_div(alpha_0, beta_0, alpha_1, beta_1)
    th_kl = kl_divergence(Gamma(alpha_0, beta_0), Gamma(alpha_1, beta_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Gamma_kl"
    ####################################################################################################################

    ####################################################################################################################
    # test multivariate_normal_kl_div
    a, b = th.randn(K, D, D), th.randn(K, D, D)
    mu_0, Sigma_0_diag, Sigma_0 = th.rand(K, D), th.rand(K, D), th.einsum('bij,bkj->bik', a, a)
    Sigma_0_stable = 5*  th.diag_embed(th.ones(K, D)) + Sigma_0
    mu_1, Sigma_1_diag, Sigma_1 = th.rand(K, D), th.rand(K, D), th.einsum('bij,bkj->bik', b, b)
    Sigma_1_stable = 5 * th.diag_embed(th.ones(K, D)) + Sigma_1

    # test diagoanl
    my_kl = mv_normal_kl_div(mu_0, Sigma_0_diag, mu_1, Sigma_1_diag, are_both_Sigma_diagonals=True)
    th_kl = kl_divergence(MultivariateNormal(mu_0, covariance_matrix=th.diag_embed(Sigma_0_diag)),
                             MultivariateNormal(mu_1, covariance_matrix=th.diag_embed(Sigma_1_diag)))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Gaussian_kl with diagonal cov matrices"

    # test one diagonal and one full
    my_kl = mv_normal_kl_div(mu_0, Sigma_0, mu_1, Sigma_1_diag, is_Sigma1_diagonal=True)
    th_kl = kl_divergence(MultivariateNormal(mu_0, covariance_matrix=Sigma_0),
                             MultivariateNormal(mu_1, covariance_matrix=th.diag_embed(Sigma_1_diag)))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Gaussian_kl with Sigma1 diagonal "

    # test full with stable matrix
    my_kl = mv_normal_kl_div(mu_0, Sigma_0_stable, mu_1, Sigma_1_stable)
    th_kl = kl_divergence(MultivariateNormal(mu_0, covariance_matrix=Sigma_0_stable),
                             MultivariateNormal(mu_1, covariance_matrix=Sigma_1_stable))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Gaussian_kl with full stable cov matrices"

    # test full with unstable matrix -> THIS TEST FAILS DUE TO NUMERICAL ERRORS
    #my_kl = multivariate_normal_kl_div(mu_0, Sigma_0, mu_1, Sigma_1)
    #th_kl = kl_divergence(MultivariateNormal(mu_0, covariance_matrix=Sigma_0),
    #                         MultivariateNormal(mu_1, covariance_matrix=Sigma_1))
    #assert th.isclose(my_kl.sum(), th_kl.sum()), "Gaussians_kl with full cov matrices"
    ####################################################################################################################

    ####################################################################################################################
    # test wishart_kl_div without batch
    a, b = th.randn(D, D), th.randn(D, D)
    n_0, V_0_diag, V_0 = D + th.randint(D, size=[1]).float(), th.rand(D), th.einsum('ij,kj->ik', a, a)
    V_0_stable = 5 * th.diag_embed(th.ones(D)) + V_0
    n_1, V_1_diag, V_1 = D + th.randint(D, size=[1]).float(), th.rand(D), th.einsum('ij,kj->ik', b, b)
    V_1_stable = 5 * th.diag_embed(th.ones(D)) + V_1

    # test diagoanl
    my_kl = wishart_kl_div(n_0, V_0_diag, n_1, V_1_diag, are_both_V_diagonals=True)
    th_kl = kl_divergence(Wishart(n_0, covariance_matrix=th.diag_embed(V_0_diag)),
                             Wishart(n_1, covariance_matrix=th.diag_embed(V_1_diag)))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Wishart_kl with diagonal matrices and no batch"

    # test one diagonal and one full
    my_kl = wishart_kl_div(n_0, V_0, n_1, V_1_diag, is_V1_diagonal=True)
    th_kl = kl_divergence(Wishart(n_0, covariance_matrix=V_0),
                             Wishart(n_1, covariance_matrix=th.diag_embed(V_1_diag)))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Wishart_kl with V1 diagonal and no batch"

    # test full with stable matrix
    my_kl = wishart_kl_div(n_0, V_0_stable, n_1, V_1_stable)
    th_kl = kl_divergence(Wishart(n_0, covariance_matrix=V_0_stable),
                             Wishart(n_1, covariance_matrix=V_1_stable))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Wishart_kl with full stable matrices and no batch"

    # test full with unstable matrix -> this might fail due to numerical errors
    my_kl = wishart_kl_div(n_0, V_0, n_1, V_1)
    th_kl = kl_divergence(Wishart(n_0, covariance_matrix=V_0),
                             Wishart(n_1, covariance_matrix=V_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Wishart_kl with full matrices and no batch"
    ####################################################################################################################
    
    # test wishart_kl when the inputs are batched -> WE BELIEVE THAT THERE IS A BUG IN PYth
    my_kl_nobatch = wishart_kl_div(n_0, V_0_diag, n_1, V_1_diag, are_both_V_diagonals=True)
    th_kl_nobatch = kl_divergence(Wishart(n_0, covariance_matrix=th.diag_embed(V_0_diag)),
                                     Wishart(n_1, covariance_matrix=th.diag_embed(V_1_diag)))
    assert th.isclose(my_kl_nobatch, th_kl_nobatch)

    # we copy the previous inputs 2 times
    n_0, V_0_diag = th.concat([n_0, n_0], dim=0), th.stack([V_0_diag, V_0_diag], dim=0)
    n_1, V_1_diag = th.concat([n_1, n_1], dim=0), th.stack([V_1_diag, V_1_diag], dim=0)

    # we compute the kl
    my_kl = wishart_kl_div(n_0, V_0_diag, n_1, V_1_diag, are_both_V_diagonals=True)
    th_kl = kl_divergence(Wishart(n_0, covariance_matrix=th.diag_embed(V_0_diag)),
                             Wishart(n_1, covariance_matrix=th.diag_embed(V_1_diag)))

    # the two vlues in my_kl and th_kl must be the same and must be equal to the result without batch
    assert th.isclose(my_kl[0], my_kl[1]) and th.isclose(my_kl[0], my_kl_nobatch), \
        "Wishart_kl with batch with same params"
    # THE FOLLOWING FAILS
    assert th.isclose(th_kl[0], th_kl[1]) and th.isclose(th_kl[0], my_kl_nobatch),\
        "pytorch Wishart_kl with batch with same params"
