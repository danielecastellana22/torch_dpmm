import torch as th
from torch_dpmm.exp_family import ExponentialFamilyDistribution

__all__ = ['BayesianDistribution']


class BayesianDistribution:
    r"""
    This class implements bayesian distributions by using conjugate priors.
    Let P(x | phi) the distribution of interest, this class define a distribution Q(phi | eta) over the parameter phi.
    Q is the conjugate prior of P and eta are the variational parameters. We represent Q using the exponential family
    representation to take advantege of the natural gradient for the SVB updates.
    """

    _exp_distr_class: ExponentialFamilyDistribution = None

    @classmethod
    def validate_common_params(cls, K: int, D: int, theta: list[th.Tensor]) -> list[th.Tensor]:
        return cls._exp_distr_class.validate_common_params(K, D, theta)

    @classmethod
    def expected_log_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        r"""
        Let Q(phi | eta), where Q is the conjugate prior distribution of P.
        This method returns the expected log value of phi w.r.t. the prior Q: E_Q[log phi].
        The computation is batched.

        Args:
            eta (list[th.Tensor]): natural params of a prior distribution. It is a list tensor K x ?, where K is the
                number of distribution that we consider.

        Returns:
            list[th.Tensor]: the expected log params
        """

        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def expected_params(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        r"""
        Let Q(phi | eta), where Q is the conjugate prior distribution of P.
        This method returns the expected value of phi w.r.t. the prior Q: E_Q[phi].
        The computation is batched.

        Args:
            eta (list[th.Tensor]): natural params of a prior distribution. It is a list tensor K x ?, where K is the
                number of distribution that we consider.

        Returns:
            list[th.Tensor]: the expected params
        """

        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def expected_data_loglikelihood(cls, obs_data: th.Tensor, eta: list[th.Tensor]) -> th.Tensor:
        r"""
        Let data ~ P(x | phi) and Q(phi | eta), where Q is the conjugate prior distribution of P.
        This method returns the expected loglikelihood of the data w.r.t. the prior Q: E_Q[log P(data)].
        The computation is batched.

        Args:
            obs_data (th.Tensor): the data observed. It is a tensor BS x D,
                BS is the batch size and D is the event shape.
            eta (list[th.Tensor]): natural params of a prior distribution. It is a list tensor K x ?, where K is the
                number of distribution that we consider.

        Returns:
            th.Tensor: the expected loglikelihood of data
        """

        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def compute_posterior_suff_stats(cls, assignments: th.Tensor, obs_data: th.Tensor) -> list[th.Tensor]:
        """
        Compute the sufficient statistics of the posterior given the observed data.
        The assignments are needed since we consider K distribution togethers to speed up the computation.
        Args:
            assignments (th.Tensor): the assignment of the observed data to the K distributions.
            obs_data (th.Tensor): the observed data.

        Returns:
            list[th.Tensor]: a list containing the sufficient statistics for the natural parameters' posterior.

        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def kl_div(cls, p_eta: list[th.Tensor], q_eta: list[th.Tensor]) -> th.Tensor:
        return cls._exp_distr_class.kl_div(p_eta, q_eta)

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        return cls._exp_distr_class.common_to_natural(theta)

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        return cls._exp_distr_class.natural_to_common(eta)


