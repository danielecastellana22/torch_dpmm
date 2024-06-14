from abc import ABC
import torch as th
from .__base__ import ExponentialFamilyDistribution
from ..misc import *
from ..constants import *


class BaseNormalInverseWishart(ExponentialFamilyDistribution, ABC):

    # We follow https://arxiv.org/pdf/2405.16088v1
    # x = mu, Sigma
    # x ~ NIW(mu_0, lambda, Phi, nu)
    # common params: mu_0, lambda, Phi, nu
    # natural params: eta_1, eta_2, eta_3, eta_4

    @classmethod
    def __det__(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __log_det__(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __inv_M__(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __inv_M_v__(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __vT_inv_M_v__(cls, M, v):
        inv_M_v = cls.__inv_M_v__(M, v)
        return th.sum(v * inv_M_v, -1)

    @classmethod
    def __inv_M_v_vT__(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __trace_M__(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def __h_x__(cls, x: list[th.Tensor]) -> th.Tensor:
        mu, Sigma = x
        D = mu.shape[-1]
        return th.pow(cls.__det__(Sigma), -0.5*(D+2))

    @classmethod
    def __A_eta__(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu0, lam, Phi, nu = cls.natural_to_common(eta)
        D = mu0.shape[-1]
        res = -0.5 * D * th.log(lam)
        res += -0.5 * nu * cls.__log_det__(Phi)
        res += 0.5 * D * LOG_2PI
        res += 0.5 * D * nu * LOG_2
        res += th.special.multigammaln(0.5 * nu, D)
        return res

    @classmethod
    def __T_x__(cls, x: list[th.Tensor], idx=None):
        mu, Sigma = x
        if idx == 0:
            return cls.__inv_M_v__(Sigma, mu)
        elif idx == 1:
            return -0.5 * cls.__vT_inv_M_v__(Sigma, mu)
        elif idx == 2:
            return -0.5 * cls.__inv_M__(Sigma)
        elif idx == 3:
            return -0.5 * cls.__log_det__(Sigma)
        else:
            return [cls.__inv_M_v__(Sigma, mu),
                    -0.5 * cls.__vT_inv_M_v__(Sigma, mu),
                    -0.5 * cls.__inv_M__(Sigma),
                    -0.5 * cls.__log_det__(Sigma)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor]:
        mu0, lam, Phi, nu = cls.natural_to_common(eta)
        D = mu0.shape[-1]
        if idx == 0:
            return nu.unsqueeze(-1) * cls.__inv_M_v__(Phi, mu0)
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * nu * cls.__trace_M__(cls.__inv_M_v_vT__(Phi, mu0))
        elif idx == 2:
            inv = cls.__inv_M__(Phi)
            return -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * cls.__inv_M__(Phi)
        elif idx == 3:
            return -0.5 * cls.__log_det__(Phi) + 0.5*D*LOG_2 + 0.5 * multidigamma(0.5 * nu, D)
        else:
            inv = cls.__inv_M__(Phi)
            return [nu.unsqueeze(-1) * cls.__inv_M_v__(Phi, mu0),
                    -0.5 * D/lam - 0.5 * nu * cls.__trace_M__(cls.__inv_M_v_vT__(Phi, mu0)),
                    -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * cls.__inv_M__(Phi),
                    -0.5 * cls.__log_det__(Phi) + 0.5*D*LOG_2 + 0.5 * multidigamma(0.5 * nu, D)]


class FullNormalInverseWishart(BaseNormalInverseWishart):

    @classmethod
    def __det__(cls, M):
        return th.det(M)

    @classmethod
    def __log_det__(cls, M):
        return th.logdet(M)

    @classmethod
    def __inv_M__(cls, M):
        return th.linalg.inv(M)

    @classmethod
    def __inv_M_v__(cls, M, v):
        return th.linalg.solve(M, v)

    @classmethod
    def __inv_M_v_vT__(cls, M, v):
        v_vT = batch_outer_product(v, v)
        return th.linalg.solve(M, v_vT)

    @classmethod
    def __trace_M__(cls, M):
        return batched_trace_square_mat(M)

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        Phi = eta_3 - lam.view(lam.shape + (1, 1)) * batch_outer_product(mu0, mu0)
        return [mu0, lam, Phi, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, Phi, nu = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = Phi + lam.view(lam.shape + (1, 1))*batch_outer_product(mu0, mu0)
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


class DiagonalNormalInverseWishart(BaseNormalInverseWishart):

    @classmethod
    def __det__(cls, M):
        return th.prod(M, -1)

    @classmethod
    def __log_det__(cls, M):
        return th.sum(th.log(M), -1)

    @classmethod
    def __inv_M__(cls, M):
        return 1/M

    @classmethod
    def __inv_M_v__(cls, M, v):
        return v / M

    @classmethod
    def __inv_M_v_vT__(cls, M, v):
        return v**2 / M

    @classmethod
    def __trace_M__(cls, M):
        return th.sum(M, -1)

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        Phi = eta_3 - lam.unsqueeze(-1) * mu0**2
        return [mu0, lam, Phi, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, Phi, nu = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = Phi + lam.unsqueeze(-1) * mu0**2
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


#TODO: implement cholesky parametrization of full covariance matrix to improve stability

if __name__ == '__main__':
    B, K, D = 100, 20, 4

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

    ####################################################################################################################
    # test multivariate_normal_kl_div
    a, b = th.randn(K, D, D) + 3*th.diag_embed(th.ones(K, D)), th.randn(K, D, D) + 3*th.diag_embed(th.ones(K, D))
    mu_0, lam_0, Phi_0_diag, Phi_0, nu_0 = (th.rand(K, D), th.rand(K), th.rand(K, D), th.einsum('bij,bkj->bik', a, a),
                                            D + th.randint(2, D, [K]))
    mu_1, lam_1, Phi_1_diag, Phi_1, nu_1 = (th.rand(K, D), th.rand(K), th.rand(K, D), th.einsum('bij,bkj->bik', a, a),
                                            D + th.randint(2, D, [K]))

    # test diagoanl
    my_kl2 = mv_normal_wishart_kl_div(mu_0, lam_0, nu_0, 1. / Phi_0_diag, mu_1, lam_1, nu_1, 1. / Phi_1_diag,
                                      is_V1_diagonal=True, is_V0_diagonal=True)
    my_kl = DiagonalNormalInverseWishart.kl_div(
        DiagonalNormalInverseWishart.common_to_natural([mu_0, lam_0, Phi_0_diag, nu_0]),
        DiagonalNormalInverseWishart.common_to_natural([mu_1, lam_1, Phi_1_diag, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with diagonal cov matrices"

    # test full
    my_kl2 = mv_normal_wishart_kl_div(mu_0, lam_0, nu_0, th.linalg.inv(Phi_0), mu_1, lam_1, nu_1, th.linalg.inv(Phi_1),
                                      is_V1_diagonal=False, is_V0_diagonal=False)
    my_kl = FullNormalInverseWishart.kl_div(
        FullNormalInverseWishart.common_to_natural([mu_0, lam_0, Phi_0, nu_0]),
        FullNormalInverseWishart.common_to_natural([mu_1, lam_1, Phi_1, nu_1]))

    assert th.isclose(my_kl.sum(), my_kl2.sum()), "mv_normal_inverse_wishart_kl with full cov matrices"

