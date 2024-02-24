import torch as th
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from ..prob_utils import *
from ..prob_utils.constants import *
from ..prob_utils.kl_div import *
from torch.special import digamma
from torch.linalg import solve, slogdet


# computes E_q [log \pi]
def E_log_pi(u, v):
    common_factor = digamma(u + v)
    expected_log_pi = digamma(u) - common_factor  # E_q[log s] (where s is the stick portion)

    aux = digamma(v) - common_factor
    expected_log_one_minus_pi = th.cumsum(aux, dim=0) - aux

    return expected_log_pi + expected_log_one_minus_pi


# computes E_q [log prec] where prec followd a Wishart distribution
def E_log_prec(n, B_inv, is_diagonal):
    D = B_inv.shape[-1]
    log_det = th.log(B_inv).sum(-1) if is_diagonal else slogdet(B_inv)[1]
    return multidigamma(0.5 * n, D) + D * LOG_2 + log_det


# computes E_q [ log N(x \mid mu, prec^-1]
def E_log_x(x, tau, c, n, B_inv, is_diagonal):
    BS, D = x.shape
    # do the broadcast to consider BS dimension
    diff = x.unsqueeze(1)-tau.unsqueeze(0)  # has shape BS x K x D
    B_inv = B_inv.unsqueeze(0).expand((BS,) + B_inv.shape)  # has shape BS x K x ?
    n = n.unsqueeze(0)
    c = c.unsqueeze(0)
    # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB) (x-mu) = D/c + n (x-mu)^T (nB) (x-mu)
    E_mahalanobis_dist = D/c + n * batch_mahalanobis_dist(B_inv, diff, is_diagonal)
    return 0.5 * (- D * LOG_2PI + E_log_prec(n, B_inv, is_diagonal) - E_mahalanobis_dist)


def natural_to_common(nat_u, nat_v, nat_tau, nat_c, nat_n, nat_B, is_diagonal):
    u = nat_u
    v = nat_v
    c = nat_c
    n = nat_n

    tau = nat_tau / nat_c.unsqueeze(-1)
    if is_diagonal:
        B = nat_B - nat_tau**2 / nat_c.view(-1, 1)
    else:
        B = nat_B - batch_outer_product(nat_tau, nat_tau) / nat_c.view(-1, 1, 1)

    return u, v, tau, c, n, B


def common_to_natural(u, v, tau, c, n, B, is_diagonal):
    ''' Convert current posterior params from common to nat form
    '''
    nat_u = u
    nat_v = v
    nat_c = c
    nat_n = n
    nat_tau = tau * c.unsqueeze(-1)

    if is_diagonal:
        nat_B = B + tau**2 * c.view(-1, 1)
    else:
        nat_B = B + batch_outer_product(tau, tau) * c.view(-1, 1, 1)

    return nat_u, nat_v, nat_tau, nat_c, nat_n, nat_B


# We use a closure to cache values of priors that do not change during the execution: e.g. the prior params and
# c0*tau0^2, B0_inv, c0*tau0
class GaussianDPMMFunctionGenerator:

    def __init__(self, alphaDP, tau0, c0, n0, B0, is_B_diagonal, is_B0_diagonal):
        K = tau0.shape[0]
        self.alphaDP = alphaDP.expand(K)  # we expand it to match the shape during the computations
        self.tau0 = tau0
        self.c0 = c0
        self.n0 = n0
        self.is_B_diagonal = is_B_diagonal
        self.is_B0_diagonal = is_B0_diagonal

        # reshape B0 in order to match the shape of B
        if is_B_diagonal == is_B0_diagonal:
            self.B0 = B0
        else:
            self.B0 = th.diag_embed(B0)
            # the case when B is diagonal and B0 is not is impossible

        # precompute and cache some values
        self.c0_tau0 = c0.unsqueeze(-1) * tau0
        tau0_square = tau0 ** 2 if is_B_diagonal else batch_outer_product(tau0, tau0)
        self.c0_tau0_2 = th.einsum("k,k...->k...", c0, tau0_square)
        self.B0_inv = 1 / B0 if is_B0_diagonal else th.linalg.inv(B0)
        self.K_ones = th.tensor(1.).expand(K)  # this is used for the beta kl

    def get_function(self):

        class GaussianDPMMFunction(Function):

            @staticmethod
            def __dpmm_computation__(x, u, v, tau, c, n, B_inv):
                pi_contribution = E_log_pi(u, v)
                data_contribution = E_log_x(x, tau, c, n, B_inv, self.is_B_diagonal)

                log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
                log_r, _ = log_normalise(log_unnorm_r)
                r = th.exp(log_r)

                # compute the elbo
                elbo = ((r * data_contribution).sum()
                        - beta_kl_div(u, v, self.K_ones, self.alphaDP).sum()
                        - mv_normal_wishart_kl_div(tau, c, n, B_inv, self.tau0, self.c0, self.n0, self.B0_inv,
                                                   is_V0_diagonal=self.is_B_diagonal,
                                                   is_V1_diagonal=self.is_B0_diagonal).sum()
                        + (r.sum(0) * pi_contribution).sum() - (
                                    r * log_r).sum())  # KL(q(z) || p(z)) where z is the cluster assignment

                return r, elbo / th.numel(x)

            @staticmethod
            def __dpmm_natural_update__(x, r):
                # The natural gradient is related to the full_batch update.
                # At first, we compute the update for all var params
                N_k = th.sum(r, dim=0)  # has shape K
                nat_u_update = (1 + N_k)
                nat_v_update = self.alphaDP + (th.sum(N_k) - th.cumsum(N_k, 0))
                nat_c_update = self.c0 + N_k
                nat_n_update = self.n0 + N_k

                nat_tau_update = self.c0_tau0 + th.matmul(r.T, x)

                x_square = x ** 2 if self.is_B_diagonal else batch_outer_product(x, x)
                r_x_square = th.einsum("bk,bk...->k...", r, x_square.unsqueeze(1))
                nat_B_update = self.B0 + self.c0_tau0_2 + r_x_square
                nat_updates = nat_u_update, nat_v_update, nat_tau_update, nat_c_update, nat_n_update, nat_B_update

                # compute mask to set gradient to 0 for components without data
                component_mask = N_k > 1e-3  # TODO: how do we choose the treshold?

                return nat_updates, component_mask

            @staticmethod
            def forward(ctx, x, *nat_params):
                u, v, tau, c, n, B = natural_to_common(*nat_params, self.is_B_diagonal)
                B_inv = 1 / B if self.is_B_diagonal else th.linalg.inv(B)
                r, elbo = GaussianDPMMFunction.__dpmm_computation__(x, u, v, tau, c, n, B_inv)
                ctx.save_for_backward(x, r, *nat_params)
                return r, -elbo

            @staticmethod
            @once_differentiable
            def backward(ctx, pi_grad, elbo_grad):
                x, r, *nat_params = ctx.saved_tensors

                nat_updates, component_mask = GaussianDPMMFunction.__dpmm_natural_update__(x, r)
                # The natural gradient is the difference between the current value and the new one
                # We also consider elbo_grad to mimic the backpropagation. It should be always 1.

                nat_params_grad = []
                for i in range(len(nat_params)):
                    nat_p = nat_params[i]
                    nat_upd = nat_updates[i]
                    grad_mask = component_mask.view(*([-1]+[1]*(nat_p.ndim-1)))
                    # mask the gradient for compoents wihtout data
                    nat_params_grad.append(elbo_grad * grad_mask *(nat_p-nat_upd))

                # there is no gradient for the data x (for now)
                x_grad = None
                return (x_grad,) + tuple(nat_params_grad)

        return GaussianDPMMFunction


