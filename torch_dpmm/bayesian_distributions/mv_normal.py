import torch as th
from .base import BayesianDistribution
from torch_dpmm.exp_family import FullNIW, DiagonalNIW, SingleNIW, SphericalNormal
from torch_dpmm.utils.constants import *
from torch_dpmm.utils.mat_utils import *


__all__ = ['FullNormalINIW', 'DiagonalNormalNIW', 'SingleNormalNIW', 'UnitNormalSpherical']


def _expand_and_compute_diff(obs_data, tau, c, B, n):
    BS, D = obs_data.shape
    # do the broadcast to consider BS dimension
    diff = obs_data.unsqueeze(1) - tau.unsqueeze(0)  # has shape BS x K x D
    B = B.unsqueeze(0).expand((BS,) + B.shape)  # has shape BS x K x D x D
    n = n.unsqueeze(0)
    c = c.unsqueeze(0)
    return diff, c, B, n


class FullNormalINIW(BayesianDistribution):
    """
    This class represent a full multivariate normal with NormalInverseWishart distribution as a prior.
    P(x | mu, Sigma) = N(mu, Sigma)
    Q(mu, Sigma | mu0, lam, nu, Phi) = NIW(mu0, lam, nu, Phi)
    """

    _exp_distr_class = FullNIW

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        tau, c, B, n = cls.natural_to_common(eta)
        D = tau.shape[-1]
        exp_Sigma = B / (n - D - 1).view(-1, 1, 1)

        return [tau, exp_Sigma]

    @classmethod
    def compute_posterior_suff_stats(cls, assignments, obs_data) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        cov_obs_data = th.einsum("bk,bk...->k...",
                                 assignments, FullMatOps.v_vT(obs_data).unsqueeze(1))
        return [mean_obs_data, N_k, cov_obs_data, N_k]

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_distr_class = cls._exp_distr_class
        D = obs_data.shape[-1]
        diff, c, B, n = _expand_and_compute_diff(obs_data, *exp_distr_class.natural_to_common(eta))
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB^-1) (x-mu) = D/c + n (x-mu)^T (B^-1) (x-mu)
        E_mahalanobis_dist = 0.5 * (D / c + n * FullMatOps.vT_inv_M_v(B, diff))
        return 0.5 * (- D * LOG_2PI) + exp_distr_class.expected_T_x(eta, 3) - E_mahalanobis_dist


class DiagonalNormalNIW(BayesianDistribution):
    r"""
    This class represent a diagonal multivariate normal with NormalInverseWishart distribution as a prior.
    P(x | mu, diag(Sigma)) = N(mu, diag(Sigma))
    Q(mu, diag(Sigma) | mu0, lam, nu, diag(Phi)) = NIW(mu0, lam, nu, diag(Phi))
    """

    _exp_distr_class = DiagonalNIW

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        tau, c, B, n = cls.natural_to_common(eta)
        D = tau.shape[-1]
        exp_Sigma = th.diag_embed(B) / (n - D - 1).view(-1, 1, 1)

        return [tau, exp_Sigma]

    @classmethod
    def compute_posterior_suff_stats(cls, assignments, obs_data) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        cov_obs_data = th.einsum("bk,bk...->k...", assignments, DiagonalMatOps.v_vT(obs_data).unsqueeze(1))
        return [mean_obs_data, N_k, cov_obs_data, N_k]

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_distr_class = cls._exp_distr_class
        D = obs_data.shape[-1]
        diff, c, B, n = _expand_and_compute_diff(obs_data, *exp_distr_class.natural_to_common(eta))
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB^-1) (x-mu) = D/c + n (x-mu)^T (B^-1) (x-mu)
        E_mahalanobis_dist = 0.5 * (D / c + n * DiagonalMatOps.vT_inv_M_v(B, diff))
        return 0.5 * (- D * LOG_2PI) + exp_distr_class.expected_T_x(eta, 3) - E_mahalanobis_dist


class SingleNormalNIW(BayesianDistribution):
    r"""
    This class represent a diagonal multivariate normal with NormalInverseWishart distribution as a prior.
    P(x | mu, s * I ) = N(mu, s * I)
    Q(mu, s | mu0, lam, nu, p) = NIW(mu0, lam, nu, p)
    """

    _exp_distr_class = SingleNIW

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        tau, c, s, n = cls.natural_to_common(eta)
        D = tau.shape[-1]
        exp_Sigma = th.diag_embed(s.view(-1, 1).expand(-1, D) / (n - 2).view(-1, 1))

        return [tau, exp_Sigma]

    @classmethod
    def compute_posterior_suff_stats(cls, assignments, obs_data) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        cov_obs_data = th.einsum("bk,bk...->k...", assignments, (obs_data**2).unsqueeze(1))
        return [mean_obs_data, N_k, cov_obs_data.mean(-1), N_k]

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_distr_class = cls._exp_distr_class
        theta = exp_distr_class.natural_to_common(eta)
        theta[2] = theta[2].unsqueeze(1) * th.ones_like(theta[0])
        diff, c, B, n = _expand_and_compute_diff(obs_data, *theta)
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB^-1) (x-mu) = D/c + n (x-mu)^T (B^-1) (x-mu)
        D = obs_data.shape[-1]
        E_mahalanobis_dist = 0.5 * (D / c + n * DiagonalMatOps.vT_inv_M_v(B, diff))
        return 0.5 * (- D * LOG_2PI) + exp_distr_class.expected_T_x(eta, 3) - E_mahalanobis_dist


class UnitNormalSpherical(BayesianDistribution):
    r"""
    This class represent a multivariate normal with identity covariance matrix with a prior only the mean.
    P(x | mu) = N(mu, I)
    Q(mu | mu0, lam) = N(mu0, 1/lam I) this is spherical normal distribution
    """

    _exp_distr_class = SphericalNormal

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_distr_class = cls._exp_distr_class
        tau, lam = exp_distr_class.natural_to_common(eta)
        BS, D = obs_data.shape
        # do the broadcast to consider BS dimension
        diff = obs_data.unsqueeze(1) - tau.unsqueeze(0)  # has shape BS x K x D
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB^-1) (x-mu) = D/c + n (x-mu)^T (B^-1) (x-mu)
        E_mahalanobis_dist = 0.5 * (th.sum(diff**2, -1))
        return - 0.5 * (D * LOG_2PI) - E_mahalanobis_dist

    @classmethod
    def expected_log_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        raise ValueError('Not needed!')

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        mu, lam = cls.natural_to_common(eta)
        return [mu, th.diag_embed(th.ones_like(mu, requires_grad=False))]

    @classmethod
    def compute_posterior_suff_stats(cls, assignments: th.Tensor, obs_data: th.Tensor) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        return [mean_obs_data, N_k]