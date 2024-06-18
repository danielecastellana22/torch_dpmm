from typing import Type
import torch as th
import torch.nn as nn
from torch_dpmm.prob_utils.misc import log_normalise
from torch_dpmm.prob_utils.conjugate_priors import ConjugatePriorDistribution, StickBreakingPrior
from torch.autograd.function import once_differentiable
from torch.autograd import Function


class DPMMFunction(Function):

    @staticmethod
    def forward(ctx, data, emission_distr_class, prior_eta, *var_eta):
        mix_weights_prior_eta, emission_prior_eta = prior_eta[:2], prior_eta[2:]
        mix_weights_var_eta, emission_var_eta = var_eta[:2], var_eta[2:]
        pi_contribution = StickBreakingPrior.expected_log_params(mix_weights_var_eta)[0]
        data_contribution = emission_distr_class.expected_data_loglikelihood(data, emission_var_eta)

        log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
        log_r, _ = log_normalise(log_unnorm_r)
        r = th.exp(log_r)

        # compute the elbo
        elbo = ((r * data_contribution).sum()
                - StickBreakingPrior.kl_div(mix_weights_var_eta, mix_weights_prior_eta).sum()
                - emission_distr_class.kl_div(emission_var_eta, emission_prior_eta).sum()
                # KL(q(z) || p(z)) where z is the cluster assignment
                + (r.sum(0) * pi_contribution).sum() - (r * log_r).sum())

        ctx.save_for_backward(r, data, *var_eta)
        ctx.emission_distr_class = emission_distr_class
        ctx.prior_eta = prior_eta

        return r, elbo / th.numel(data)

    @staticmethod
    @once_differentiable
    def backward(ctx, pi_grad, elbo_grad):
        emission_distr_class = ctx.emission_distr_class
        prior_eta = ctx.prior_eta
        r, data, *var_eta = ctx.saved_tensors

        var_eta_suff_stasts = StickBreakingPrior.compute_posterior_nat_params(r) + \
                      emission_distr_class.compute_posterior_nat_params(r, data)

        var_eta_updates = [prior_eta[i] + var_eta_suff_stasts[i] for i in range(len(prior_eta))]

        # The natural gradient is the difference between the current value and the new one
        # We also consider elbo_grad to mimic the backpropagation. It should be always 1.
        var_eta_grads = [(var_eta[i] - var_eta_updates[i]) * elbo_grad for i in range(len(var_eta))]

        # there is no gradient for the data x
        return (None, None, None) + tuple(var_eta_grads)


class DPMM(nn.Module):

    def __init__(self, K, D, alphaDP,
                 emission_distr_class: Type[ConjugatePriorDistribution], emission_prior_theta: list[th.Tensor]):
        super().__init__()

        self.K = K  # number of mixture components
        self.D = D  # size of output vector
        self.emission_distr_class = emission_distr_class

        # store the prior nat params of the mixture weights and create the variational parameters
        mix_weights_prior_theta = StickBreakingPrior.validate_common_params(K, D, [1, alphaDP])
        self.mix_weights_var_eta = []
        self.mix_weights_prior_eta = []
        for i, p in enumerate(StickBreakingPrior.common_to_natural(mix_weights_prior_theta)):
            b_name = f'mix_prior_eta_{i}'
            p_name = f'mix_var_eta_{i}'
            self.register_buffer(b_name, p)
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.mix_weights_prior_eta.append(self.get_buffer(b_name))
            self.mix_weights_var_eta.append(self.get_parameter(p_name))

        emission_prior_theta = emission_distr_class.validate_common_params(K, D, emission_prior_theta)
        self.emission_var_eta = []
        self.emission_prior_eta = []
        for i, p in enumerate(emission_distr_class.common_to_natural(emission_prior_theta)):
            b_name = f'emission_prior_eta_{i}'
            p_name = f'emission_var_eta_{i}'
            self.register_buffer(b_name, p)
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.emission_prior_eta.append(self.get_buffer(b_name))
            self.emission_var_eta.append(self.get_parameter(p_name))

        self.init_var_params()

    def forward(self, x):
        return DPMMFunction.apply(x, self.emission_distr_class,
                                  self.mix_weights_prior_eta + self.emission_prior_eta,  # concatenate the eta lists
                                  *(self.mix_weights_var_eta + self.emission_var_eta))  # concatenate the eta lists

    def init_var_params(self, x=None):
        for p in self.mix_weights_var_eta:
            nn.init.ones_(p)
        init_v = self._get_init_vals_emission_var_eta(x)
        for i, p in enumerate(self.emission_var_eta):
            p.data = init_v[i]

    def _get_init_vals_emission_var_eta(self, x):
        raise NotImplementedError('This should be implmented in the sublcasses!')

    @th.no_grad()
    def get_var_params(self):
        params = StickBreakingPrior.natural_to_common(self.mix_weights_var_eta) + \
                 self.emission_distr_class.natural_to_common(self.emission_var_eta)

        return (p.detach() for p in params)

    @th.no_grad()
    def get_num_active_components(self):
        r = StickBreakingPrior.expected_params(self.mix_weights_var_eta)[0]
        return th.sum(r > 0.01).item()

    @th.no_grad()
    def get_expected_params(self, return_log_r=False):
        r = StickBreakingPrior.expected_params(self.mix_weights_var_eta)[0]
        expected_emission_params = self.emission_distr_class.expected_params(self.emission_var_eta)

        return r, expected_emission_params

    @th.no_grad()
    def get_expected_log_likelihood(self, x):
        log_r = StickBreakingPrior.expected_log_params(self.mix_weights_var_eta)
        exp_data_loglike = self.emission_distr_class.expected_data_loglikelihood(x, self.emission_var_eta)
        _, logZ = log_normalise(exp_data_loglike+log_r)
        return logZ

