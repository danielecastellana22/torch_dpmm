from typing import Type
import torch as th
from .__base__ import ConjugatePriorDistribution
from ..exp_distributions import Beta, ExponentialFamilyDistribution


class StickBreakingPrior(ConjugatePriorDistribution):

    __exp_distr_class__ = Beta

    @classmethod
    def expected_data_loglikelihood(cls, obs_data, eta):
        raise NotImplementedError('To be implemented!')

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> th.Tensor:
        return th.exp(cls.expected_log_params(eta))

    @classmethod
    def expected_log_params(cls, eta):
        exp_T_x = cls.exp_family_distribution_class.expected_T_x(eta)
        expected_log_pi = exp_T_x[0]
        expected_log_one_minus_pi = th.cumsum(exp_T_x[1], dim=0) - exp_T_x[1]

        return expected_log_pi + expected_log_one_minus_pi

    @classmethod
    def compute_posterior_nat_params(cls, assignments, obs_data=None) -> list[th.Tensor]:
        assert obs_data is None
        N_k = th.sum(assignments, dim=0)  # has shape K
        return [N_k, (th.sum(N_k) - th.cumsum(N_k, 0))]
