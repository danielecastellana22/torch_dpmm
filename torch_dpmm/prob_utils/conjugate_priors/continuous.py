from abc import ABC
import torch as th
from .base import ConjugatePriorDistribution
from ..exp_distributions import FullNIW, DiagonalNIW
from ..misc import batch_outer_product
from ..constants import *


class BaseNIWPrior(ConjugatePriorDistribution, ABC):

    _theta_names = ['mu0', 'lmabda', 'Phi', 'nu']

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        exp_distr_class = cls._exp_distr_class
        tau, c, B, n = exp_distr_class.natural_to_common(eta)
        BS, D = obs_data.shape
        # do the broadcast to consider BS dimension
        diff = obs_data.unsqueeze(1) - tau.unsqueeze(0)  # has shape BS x K x D
        B = B.unsqueeze(0).expand((BS,) + B.shape)  # has shape BS x K x D x D
        n = n.unsqueeze(0)
        c = c.unsqueeze(0)
        # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB) (x-mu) = D/c + n (x-mu)^T (nB) (x-mu)
        E_mahalanobis_dist = 0.5 * (D / c + n * exp_distr_class._vT_inv_M_v(B, diff))
        return 0.5 * (- D * LOG_2PI) + exp_distr_class.expected_T_x(eta, 3) - E_mahalanobis_dist

    def expected_log_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        raise ValueError('Not needed!')


class FullINIWPrior(BaseNIWPrior):

    _exp_distr_class = FullNIW
    _theta_shape_list = ['[K, D]', '[K]', '[K, D, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'PositiveDefinite()', 'GreaterThan(D+1)']

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        exp_family_distr_class = cls._exp_distr_class
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


class DiagonalNIWPrior(BaseNIWPrior):

    _exp_distr_class = DiagonalNIW
    _theta_shape_list = ['[K, D]', '[K]', '[K, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'Positive()', 'GreaterThan(D+1)']

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        exp_family_distr_class = cls._exp_distr_class
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
