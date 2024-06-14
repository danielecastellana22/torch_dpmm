from abc import ABC
import torch as th
from .__base__ import ConjugatePriorDistribution
from ..exp_distributions import FullNormalInverseWishart, DiagonalNormalInverseWishart
from ..constants import *
from ..misc import batch_outer_product


class BaseNormalInverseWishartPrior(ConjugatePriorDistribution, ABC):

    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_family_distr_class = cls.__exp_distr_class__
        tau, c, B, n = exp_family_distr_class.natural_to_common(eta)
        BS, D = obs_data.shape
        # do the broadcast to consider BS dimension
        diff = obs_data.unsqueeze(1) - tau.unsqueeze(0)  # has shape BS x K x D
        B = B.unsqueeze(0).expand((BS,) + B.shape)  # has shape BS x K x D x D
        n = n.unsqueeze(0)
        c = c.unsqueeze(0)
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB) (x-mu) = D/c + n (x-mu)^T (nB) (x-mu)
        E_mahalanobis_dist = D / c + n * exp_family_distr_class.__vT_inv_M_v__(B, diff)
        return 0.5 * (- D * LOG_2PI + exp_family_distr_class.expected_T_x(eta, 3) - E_mahalanobis_dist)

    def expected_log_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        raise ValueError('Not needed!')


class FullINormalInverseWishartPrior(BaseNormalInverseWishartPrior):

    __exp_distr_class__ = FullNormalInverseWishart

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        exp_family_distr_class = cls.exp_family_distribution_class
        tau, c, B, n = exp_family_distr_class.natural_to_common(eta)
        D = tau.shape[-1]
        exp_Sigma = B / (n - D - 1).view(-1, 1, 1)

        return [tau, exp_Sigma]

    @classmethod
    def compute_posterior_nat_params(cls, assignments, obs_data) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        cov_obs_data = th.einsum("bk,bk...->k...",
                                 assignments, batch_outer_product(obs_data, obs_data).unsqueeze(1))
        return [mean_obs_data, N_k, cov_obs_data, N_k]


class DiagonalNormalInversePrior(BaseNormalInverseWishartPrior):

    __exp_distr_class__ = DiagonalNormalInverseWishart

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        exp_family_distr_class = cls.exp_family_distribution_class
        tau, c, B, n = exp_family_distr_class.natural_to_common(eta)
        D = tau.shape[-1]
        exp_Sigma = th.diag_embed(B) / (n - D - 1).view(-1, 1, 1)

        return [tau, exp_Sigma]

    @classmethod
    def compute_posterior_nat_params(cls, assignments, obs_data) -> list[th.Tensor]:
        N_k = th.sum(assignments, dim=0)  # has shape K
        mean_obs_data = th.matmul(assignments.T, obs_data)
        cov_obs_data = th.einsum("bk,bk...->k...", assignments, (obs_data**2).unsqueeze(1))
        return [mean_obs_data, N_k, cov_obs_data, N_k]

# TODO: add cholesky parametrisation to improve stability
