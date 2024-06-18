from abc import ABC
import torch as th
from .base import ExponentialFamilyDistribution
from ..misc import *
from ..constants import *


class BaseNIW(ExponentialFamilyDistribution, ABC):

    # We follow https://arxiv.org/pdf/2405.16088v1
    # x = mu, Sigma
    # x ~ NIW(mu_0, lambda, Phi, nu)
    # common params: mu_0, lambda, Phi, nu
    # natural params: eta_1, eta_2, eta_3, eta_4

    @classmethod
    def _det(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _log_det(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _inv_M(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _inv_M_v(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _vT_inv_M_v(cls, M, v):
        inv_M_v = cls._inv_M_v(M, v)
        return th.sum(v * inv_M_v, -1)

    @classmethod
    def _inv_M_v_vT(cls, M, v):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _trace_M(cls, M):
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        mu, Sigma = x
        D = mu.shape[-1]
        return th.pow(cls._det(Sigma), -0.5 * (D + 2))

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu0, lam, Phi, nu = cls.natural_to_common(eta)
        D = mu0.shape[-1]
        res = -0.5 * D * th.log(lam)
        res += -0.5 * nu * cls._log_det(Phi)
        res += 0.5 * D * LOG_2PI
        res += 0.5 * D * nu * LOG_2
        res += th.special.multigammaln(0.5 * nu, D)
        return res

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None):
        mu, Sigma = x
        if idx == 0:
            return cls._inv_M_v(Sigma, mu)
        elif idx == 1:
            return -0.5 * cls._vT_inv_M_v(Sigma, mu)
        elif idx == 2:
            return -0.5 * cls._inv_M(Sigma)
        elif idx == 3:
            return -0.5 * cls._log_det(Sigma)
        else:
            return [cls._inv_M_v(Sigma, mu),
                    -0.5 * cls._vT_inv_M_v(Sigma, mu),
                    -0.5 * cls._inv_M(Sigma),
                    -0.5 * cls._log_det(Sigma)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor]:
        mu0, lam, Phi, nu = cls.natural_to_common(eta)
        D = mu0.shape[-1]
        if idx == 0:
            return nu.unsqueeze(-1) * cls._inv_M_v(Phi, mu0)
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * nu * cls._trace_M(cls._inv_M_v_vT(Phi, mu0))
        elif idx == 2:
            inv = cls._inv_M(Phi)
            return -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv
        elif idx == 3:
            return -0.5 * cls._log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)
        else:
            inv = cls._inv_M(Phi)
            return [nu.unsqueeze(-1) * cls._inv_M_v(Phi, mu0),
                    -0.5 * D / lam - 0.5 * nu * cls._trace_M(cls._inv_M_v_vT(Phi, mu0)),
                    -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv,
                    -0.5 * cls._log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)]


class FullNIW(BaseNIW):

    @classmethod
    def _det(cls, M):
        return th.det(M)

    @classmethod
    def _log_det(cls, M):
        return th.logdet(M)

    @classmethod
    def _inv_M(cls, M):
        return th.linalg.inv(M)

    @classmethod
    def _inv_M_v(cls, M, v):
        return th.linalg.solve(M, v)

    @classmethod
    def _inv_M_v_vT(cls, M, v):
        v_vT = batch_outer_product(v, v)
        return th.linalg.solve(M, v_vT)

    @classmethod
    def _trace_M(cls, M):
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


class DiagonalNIW(BaseNIW):

    @classmethod
    def _det(cls, M):
        return th.prod(M, -1)

    @classmethod
    def _log_det(cls, M):
        return th.sum(th.log(M), -1)

    @classmethod
    def _inv_M(cls, M):
        return 1/M

    @classmethod
    def _inv_M_v(cls, M, v):
        return v / M

    @classmethod
    def _inv_M_v_vT(cls, M, v):
        return v**2 / M

    @classmethod
    def _trace_M(cls, M):
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


# TODO: implement cholesky parametrization of full covariance matrix to improve stability.

# TODO: maybe its reasonable to implement also LKJ prior