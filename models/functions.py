import torch as th
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from ..prob_utils.conjugate_priors import ConjugatePriorDistribution, StickBreakingPrior
from ..prob_utils.misc import log_normalise


class DPMMFunctionGenerator:

    def __init__(self, mix_weights_prior_nat_params, emission_prior_class: ConjugatePriorDistribution, emission_prior_nat_params):
        self.mix_weights_prior_nat_params = mix_weights_prior_nat_params
        self.emission_prior_class = emission_prior_class
        self.emission_prior_nat_params = emission_prior_nat_params

    def get_function(self):

        class GaussianDPMMFunction(Function):

            @staticmethod
            def forward(ctx, data, mix_weights_var_nat_params, emission_var_nat_params):
                pi_contribution = StickBreakingPrior.expected_log_params(self.mix_weights_prior_nat_params)
                data_contribution = self.emission_prior_class.expected_data_loglikelihood(data, emission_var_nat_params)

                log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
                log_r, _ = log_normalise(log_unnorm_r)
                r = th.exp(log_r)

                # compute the elbo
                elbo = ((r * data_contribution).sum()
                        - StickBreakingPrior.kl_div(mix_weights_var_nat_params,
                                                    self.mix_weights_prior_nat_params).sum()
                        - self.emission_prior_class.kl_div(emission_var_nat_params,
                                                           self.emission_prior_nat_params).sum()
                        # KL(q(z) || p(z)) where z is the cluster assignment
                        + (r.sum(0) * pi_contribution).sum() - (r * log_r).sum())

                ctx.save_for_backward(r, data, mix_weights_var_nat_params, emission_var_nat_params)

                return r, elbo / th.numel(data)

            @staticmethod
            @once_differentiable
            def backward(ctx, pi_grad, elbo_grad):
                r, data, mix_weights_var_nat_params, emission_var_nat_params = ctx.saved_tensors

                mix_weights_var_nat_updates = StickBreakingPrior.compute_posterior_nat_params(r)
                emission_var_nat_updates = self.emission_prior_class.compute_posterior_nat_params(r, data)

                # The natural gradient is the difference between the current value and the new one
                # We also consider elbo_grad to mimic the backpropagation. It should be always 1.
                mix_weights_var_nat_grad = [(mix_weights_var_nat_params[i] - mix_weights_var_nat_updates[i]) * elbo_grad
                                            for i in range(len(mix_weights_var_nat_params))]
                emission_var_nat_grad = [(emission_var_nat_params[i] - emission_var_nat_updates[i]) * elbo_grad
                                         for i in range(len(emission_var_nat_params))]

                # there is no gradient for the data x
                return None, mix_weights_var_nat_grad, emission_var_nat_grad

        return GaussianDPMMFunction


