from typing import Type
import torch as th
from .base import ConjugatePriorDistribution
from ..exp_distributions import Beta
from ..constraints import *


class StickBreakingPrior(ConjugatePriorDistribution):

    _exp_distr_class = Beta
    _theta_names = ['aplha', 'beta']
    _theta_shape_list = ['[K]', '[K]']
    _theta_constraints_list = ['Positive()', 'Positive()']

    @classmethod
    def expected_data_loglikelihood(cls, obs_data, eta):
        raise NotImplementedError('To be implemented!')

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        return [th.exp(v) for v in cls.expected_log_params(eta)]

    @classmethod
    def expected_log_params(cls, eta) -> list[th.Tensor]:
        exp_T_x = cls._exp_distr_class.expected_T_x(eta)
        expected_log_pi = exp_T_x[0]
        expected_log_one_minus_pi = th.cumsum(exp_T_x[1], dim=0) - exp_T_x[1]

        return [expected_log_pi + expected_log_one_minus_pi]

    @classmethod
    def compute_posterior_nat_params(cls, assignments, obs_data=None) -> list[th.Tensor]:
        assert obs_data is None
        N_k = th.sum(assignments, dim=0)  # has shape K
        return [N_k, (th.sum(N_k) - th.cumsum(N_k, 0))]
