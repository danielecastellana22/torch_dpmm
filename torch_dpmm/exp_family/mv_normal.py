import torch as th
from abc import ABC
from .base import ExponentialFamilyDistribution
from torch_dpmm.utils.constants import *
from torch_dpmm.utils.mat_utils import *
from torch_dpmm.utils.misc import *

__all__ = ['FullNIW', 'DiagonalNIW', 'SingleNIW', 'SphericalNormal']


class BaseNIW(ExponentialFamilyDistribution, ABC):

    # We follow https://arxiv.org/pdf/2405.16088v1
    # x = mu, Sigma
    # x ~ NIW(mu_0, lambda, Phi, nu)
    # common params: mu_0, lambda, Phi, nu
    # natural params: eta_1, eta_2, eta_3, eta_4

    _theta_names = ['mu0', 'lambda', 'Phi', 'nu']
    _mat_ops_class = None

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        cls._inner_h_x(*x)

    @classmethod
    def _inner_h_x(cls, mu, Sigma):
        D = mu.shape[-1]
        return th.pow(cls._mat_ops_class.det(Sigma), -0.5 * (D + 2)) * th.pow(PI, -0.5 * D)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        return cls._inner_A_eta(*cls.natural_to_common(eta))

    @classmethod
    def _inner_A_eta(cls, mu0, lam, Phi, nu):
        D = mu0.shape[-1]
        res = -0.5 * D * th.log(lam)
        res += -0.5 * nu * cls._mat_ops_class.log_det(Phi)
        res += 0.5 * D * nu * LOG_2
        res += th.special.multigammaln(0.5 * nu, D)
        return res

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        return cls._inner_T_x(*x, idx)

    @classmethod
    def _inner_T_x(cls, mu, Sigma, idx):
        if idx == 0:
            return cls._mat_ops_class.inv_M_v(Sigma, mu)
        elif idx == 1:
            return -0.5 * cls._mat_ops_class.vT_inv_M_v(Sigma, mu)
        elif idx == 2:
            return -0.5 * cls._mat_ops_class.inv_M(Sigma)
        elif idx == 3:
            return -0.5 * cls._mat_ops_class.log_det(Sigma)
        else:
            return [cls._mat_ops_class.inv_M_v(Sigma, mu),
                    -0.5 * cls._mat_ops_class.vT_inv_M_v(Sigma, mu),
                    -0.5 * cls._mat_ops_class.inv_M(Sigma),
                    -0.5 * cls._mat_ops_class.log_det(Sigma)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        return cls._inner_expected_T_x(*cls.natural_to_common(eta), idx)

    @classmethod
    def _inner_expected_T_x(cls, mu0, lam, Phi, nu, idx):
        D = mu0.shape[-1]
        if idx == 0:
            return nu.unsqueeze(-1) * cls._mat_ops_class.inv_M_v(Phi, mu0)
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * nu * cls._mat_ops_class.trace_M(cls._mat_ops_class.inv_M_v_vT(Phi, mu0))
        elif idx == 2:
            inv = cls._mat_ops_class.inv_M(Phi)
            return -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv
        elif idx == 3:
            return -0.5 * cls._mat_ops_class.log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)
        else:
            inv = cls._mat_ops_class.inv_M(Phi)
            return [nu.unsqueeze(-1) * cls._mat_ops_class.inv_M_v(Phi, mu0),
                    -0.5 * D / lam - 0.5 * nu * cls._mat_ops_class.trace_M(cls._mat_ops_class.inv_M_v_vT(Phi, mu0)),
                    -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv,
                    -0.5 * cls._mat_ops_class.log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)]

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        Phi = eta_3 - lam.view(lam.shape + (1, )*(eta_3.ndim -1)) * cls._mat_ops_class.v_vT(mu0)
        return [mu0, lam, Phi, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, Phi, nu = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = Phi + lam.view(lam.shape + (1, )*(Phi.ndim -1)) * cls._mat_ops_class.v_vT(mu0)
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


class FullNIW(BaseNIW):

    _mat_ops_class = FullMatOps
    _theta_shape_list = ['[K, D]', '[K]', '[K, D, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'PositiveDefinite()', 'GreaterThan(D+1)']


class DiagonalNIW(BaseNIW):

    _mat_ops_class = DiagonalMatOps
    _theta_shape_list = ['[K, D]', '[K]', '[K, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'Positive()', 'GreaterThan(D+1)']


class SingleNIW(DiagonalNIW):

    _theta_shape_list = ['[K, D]', '[K]', '[K]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'Positive()', 'GreaterThan(2)']

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        mu, s = x[0], x[1]
        D = mu.shape[-1]
        Sigma = s.view(-1, 1).expand(-1, D)
        return cls._inner_h_x(mu, Sigma)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu, lam, p, nu = cls.natural_to_common(eta)
        D = mu.shape[-1]
        Phi = p.view(-1, 1).expand(-1, D)
        return cls._inner_A_eta(mu, lam, Phi, nu)

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, s = x[0], x[1]
        D = mu.shape[-1]
        Sigma = s.view(-1, 1).expand(-1, D)
        ris = cls._inner_T_x(mu, Sigma, idx)
        if idx is None:
            ris[2] = ris[2].sum(-1)
        elif idx == 2:
            ris = ris.sum(-1)
        return ris

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, lam, p, nu = cls.natural_to_common(eta)
        D = mu.shape[-1]
        Phi = p.view(-1, 1).expand(-1, D)
        ris = cls._inner_expected_T_x(mu, lam, Phi, nu, idx)
        if idx is None:
            ris[2] = ris[2].sum(-1)
        elif idx == 2:
            ris = ris.sum(-1)
        return ris

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        D = mu0.shape[-1]
        p = eta_3 - lam * th.sum(mu0 * mu0, -1) / D
        return [mu0, lam, p, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, p, nu = theta
        D = mu0.shape[-1]
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = p + lam * th.sum(mu0 * mu0, -1) / D
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


class SphericalNormal(ExponentialFamilyDistribution):

    # x
    # x ~ UnitNormal(mu_0, lambda) = N(mu, 1/lam*I)
    # common params: mu_0, lambda
    # natural params: eta_1, eta_2

    _theta_names = ['mu', 'lam']
    _theta_shape_list = ['[K, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()']

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        D = x[0].shape[-1]
        return th.pow(PI, -0.5 * D)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu, lam = cls.natural_to_common(eta)
        D = mu.shape[-1]
        return - 0.5 * D * th.log(lam) + 0.5 * lam * th.sum(mu**2, -1)

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        x = x[0]
        if idx == 0:
            return x
        elif idx == 1:
            return th.sum(x**2, -1)
        else:
            return [x, -0.5 * th.sum(x**2, -1)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, lam = cls.natural_to_common(eta)
        D = mu.shape[-1]
        if idx == 0:
            return mu
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * th.sum(mu**2, -1)
        else:
            return [mu, -0.5 * D/lam - 0.5 * th.sum(mu**2, -1)]

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2 = eta
        lam = eta_2
        mu0 = eta_1 / lam.unsqueeze(-1)
        return [mu0, lam]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        return [eta_1, eta_2]


# TODO: implement cholesky parametrization of full covariance matrix to improve stability.


# TODO: maybe its reasonable to implement also LKJ prior
