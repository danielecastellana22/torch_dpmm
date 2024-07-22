from typing import Type
import torch as th
import torch.nn as nn
from torch_dpmm.utils.misc import log_normalise
from torch_dpmm.bayesian_distributions import BayesianDistribution, CategoricalSBP
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from torch_dpmm import _DEBUG_MODE


class DPMMFunction(Function):

    @staticmethod
    def forward(ctx, data, emission_distr_class, prior_eta, *var_eta):
        mix_weights_prior_eta, emission_prior_eta = prior_eta[:2], prior_eta[2:]
        mix_weights_var_eta, emission_var_eta = var_eta[:2], var_eta[2:]
        pi_contribution = CategoricalSBP.expected_log_params(mix_weights_var_eta)[0]
        data_contribution = emission_distr_class.expected_data_loglikelihood(data, emission_var_eta)

        log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
        log_r, log_Z = log_normalise(log_unnorm_r)
        r = th.exp(log_r)

        # compute the elbo
        elbo = ((r * data_contribution).sum()
                - CategoricalSBP.kl_div(mix_weights_var_eta, mix_weights_prior_eta).sum()
                - emission_distr_class.kl_div(emission_var_eta, emission_prior_eta).sum()
                # KL(q(z) || p(z)) where z is the cluster assignment
                + (r.sum(0) * pi_contribution).sum() - (r * log_r).sum())

        ctx.save_for_backward(r, data, *var_eta)
        ctx.emission_distr_class = emission_distr_class
        ctx.prior_eta = prior_eta

        return r, -elbo / th.numel(data), log_Z

    @staticmethod
    @once_differentiable
    def backward(ctx, pi_grad, elbo_grad, log_Z_grad):
        emission_distr_class = ctx.emission_distr_class
        prior_eta = ctx.prior_eta
        r, data, *var_eta = ctx.saved_tensors

        var_eta_suff_stasts = CategoricalSBP.compute_posterior_suff_stats(r) + \
                              emission_distr_class.compute_posterior_suff_stats(r, data)

        var_eta_updates = [prior_eta[i] + var_eta_suff_stasts[i] for i in range(len(prior_eta))]

        if _DEBUG_MODE:
            K, D = r.shape[-1], data.shape[-1]
            sbp_updates = var_eta_updates[:2]
            emiss_updates = var_eta_updates[2:]
            CategoricalSBP.validate_common_params(K, D, CategoricalSBP.natural_to_common(sbp_updates))
            emission_distr_class.validate_common_params(K, D, emission_distr_class.natural_to_common(emiss_updates))

        # The natural gradient is the difference between the current value and the new one
        # We also consider elbo_grad to mimic the backpropagation. It should be always 1.
        var_eta_grads = [(var_eta[i] - var_eta_updates[i]) * elbo_grad for i in range(len(var_eta))]

        # there is no gradient for the data x
        return (None, None, None) + tuple(var_eta_grads)


class DPMM(nn.Module):

    def __init__(self, K, D, alphaDP,
                 emission_distr_class: Type[BayesianDistribution], emission_prior_theta: list[th.Tensor]):
        super().__init__()

        self.K = K  # number of mixture components
        self.D = D  # size of output vector
        self.emission_distr_class = emission_distr_class

        # store the prior nat params of the mixture weights and create the variational parameters
        mix_weights_prior_theta = CategoricalSBP.validate_common_params(K, D, [1, alphaDP])
        self.mix_weights_var_eta = []
        for i, p in enumerate(CategoricalSBP.common_to_natural(mix_weights_prior_theta)):
            b_name = f'mix_prior_eta_{i}'
            p_name = f'mix_var_eta_{i}'
            self.register_buffer(b_name, p.contiguous())
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.mix_weights_var_eta.append(self.get_parameter(p_name))

        emission_prior_theta = emission_distr_class.validate_common_params(K, D, emission_prior_theta)
        self.emission_var_eta = []
        for i, p in enumerate(emission_distr_class.common_to_natural(emission_prior_theta)):
            b_name = f'emission_prior_eta_{i}'
            p_name = f'emission_var_eta_{i}'
            self.register_buffer(b_name, p.contiguous())
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.emission_var_eta.append(self.get_parameter(p_name))

        self.init_var_params()

    @property
    def mix_weights_prior_eta(self):
        return [self.get_buffer(f'mix_prior_eta_{i}') for i in range(len(self.mix_weights_var_eta))]

    @property
    def emission_prior_eta(self):
        return [self.get_buffer(f'emission_prior_eta_{i}') for i in range(len(self.emission_var_eta))]

    def forward(self, x):
        return DPMMFunction.apply(x, self.emission_distr_class,
                                  self.mix_weights_prior_eta + self.emission_prior_eta,  # concatenate the eta lists
                                  *(self.mix_weights_var_eta + self.emission_var_eta))  # concatenate the eta lists

    def init_var_params(self, x=None, mask=None, mix_init_theta=None, emission_init_theta=None):
        if mask is None:
            mask = th.ones(self.K, dtype=th.bool, device=self.mix_weights_var_eta[0].device)

        K = th.sum(mask)
        mix_init_eta = None
        if mix_init_theta is not None:
            mix_init_theta = CategoricalSBP.validate_common_params(K, self.D, mix_init_theta)
            mix_init_eta = CategoricalSBP.common_to_natural(mix_init_theta)

        for i, p in enumerate(self.mix_weights_var_eta):
            if mix_init_eta is not None:
                p.data[mask] = mix_init_eta[i]
            else:
                p.data[mask] = 1

        if emission_init_theta is not None:
            emission_init_theta = self.emission_distr_class.validate_common_params(K, self.D, emission_init_theta)
            emission_init_eta = self.emission_distr_class.common_to_natural(emission_init_theta)
        else:
            emission_init_eta = self._get_init_vals_emission_var_eta(x, mask)

        for i, p in enumerate(self.emission_var_eta):
            p.data[mask] = emission_init_eta[i]

    def _get_init_vals_emission_var_eta(self, x, mask):
        raise NotImplementedError('This should be implmented in the sublcasses!')

    @th.no_grad()
    def get_var_params(self):
        params = CategoricalSBP.natural_to_common(self.mix_weights_var_eta) + \
                 self.emission_distr_class.natural_to_common(self.emission_var_eta)

        return (p.detach() for p in params)

    @th.no_grad()
    def get_num_active_components(self):
        r = CategoricalSBP.expected_params(self.mix_weights_var_eta)[0]
        return th.sum(r > 0.01).item()

    @th.no_grad()
    def get_expected_params(self):
        r = CategoricalSBP.expected_params(self.mix_weights_var_eta)[0].detach()
        expected_emission_params = [v.detach() for v in self.emission_distr_class.expected_params(self.emission_var_eta)]

        return r, expected_emission_params

