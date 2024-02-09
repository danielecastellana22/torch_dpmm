import torch as th
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from prob_utils import *
from prob_utils.constants import *
from prob_utils.kl_div import *
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
def E_log_prec(n, B, is_diagonal):
    D = B.shape[-1]
    log_det = th.log(B).sum(-1) if is_diagonal else slogdet(B)[1]
    return multidigamma(0.5 * n, D) + D * LOG_2 + log_det


# computes E_q [ log N(x \mid mu, prec^-1]
def E_log_x(x, tau, c, n, B, is_diagonal):
    BS, D = x.shape
    # do the broadcast to consider BS dimension
    diff = x.unsqueeze(1)-tau.unsqueeze(0)  # has shape BS x K x D
    B = B.unsqueeze(0).expand((BS,) + B.shape)  # has shape BS x K x ?
    n = n.unsqueeze(0)
    c = c.unsqueeze(0)
    # E_q [(x-mu)^T (cPrec)^-1 (x-mu)] = D/c + (x-mu)^T (nB) (x-mu) = D/c + n (x-mu)^T (nB) (x-mu)
    E_mahalanobis_dist = D/c + n * batch_mahalanobis_dist(B, diff, is_diagonal)
    return 0.5 * (- D * LOG_2PI + E_log_prec(n, B, is_diagonal) - E_mahalanobis_dist)


# We use a closure to cache values of priors that do not change during the execution: e.g. the prior params and
# c0*tau0^2, B0_inv, c0*tau0
class GaussianDPMMFunctionGenerator:

    def __init__(self, alphaDP, tau0, c0, n0, B0, is_B_diagonal, is_B0_diagonal):
        K = tau0.shape[0]
        self.alphaDP = alphaDP.expand(K)  # we expand it to match the shape during the computations
        self.tau0 = tau0
        self.c0 = c0
        self.n0 = n0
        self.B0 = B0
        self.is_B_diagonal = is_B_diagonal
        self.is_B0_diagonal = is_B0_diagonal

        # precompute and cache some values
        self.c0_tau0 = c0.unsqueeze(-1) * tau0
        tau0_square = tau0 ** 2 if is_B_diagonal else batch_outer_product(tau0, tau0)
        self.c0_tau0_2 = th.einsum("k,k...->k...", c0, tau0_square)
        D = B0.shape[-1]
        self.B0_inv = 1 / B0 if is_B0_diagonal else solve(B0, th.ones(D))
        self.K_ones = th.tensor(1.).expand(K)  # this is used for the beta kl

    def get_function(self):

        class GaussianDPMMFunction(Function):

            @staticmethod
            def __dpmm_computation__(x, u, v, tau, c, n, B):
                pi_contribution = E_log_pi(u, v)
                data_contribution = E_log_x(x, tau, c, n, B, self.is_B_diagonal)

                log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
                log_r, _ = log_normalise(log_unnorm_r)
                r = th.exp(log_r)

                # compute the elbo
                elbo = ((r * data_contribution).sum()
                        - beta_kl_div(u, v, self.K_ones, self.alphaDP).sum()
                        - mv_normal_wishart_kl_div(tau, c, n, B, self.tau0, self.c0, self.n0, self.B0,
                                                   is_V0_diagonal=self.is_B_diagonal,
                                                   is_V1_diagonal=self.is_B0_diagonal).sum()
                        + (r.sum(0) * pi_contribution).sum() - (
                                    r * log_r).sum())  # KL(q(z) || p(z)) where z is the cluster assignment

                return r, elbo / th.numel(x)

            @staticmethod
            def __dpmm_update__(x, r, u, v, tau, c, n, B):
                D = x.shape[-1]
                # compute the MINUS natural gradient for the variational parameters
                # the minus is because max ELBO = min -ELBO

                # The natural gradient is related to the full_batch update.
                # At first, we compute the update for all var params
                N_k = th.sum(r, dim=0)  # has shape K
                u_update = (1 + N_k)
                v_update = self.alphaDP + (th.sum(N_k) - th.cumsum(N_k, 0))
                c_update = self.c0 + N_k
                tau_update = (self.c0_tau0 + th.matmul(r.T, x)) / c_update.unsqueeze(-1)
                n_update = self.n0 + N_k

                x_square = x ** 2 if self.is_B_diagonal else batch_outer_product(x, x)
                r_x_square = th.einsum("bk,bk...->k...", r, x_square.unsqueeze(1))
                tau_square = tau ** 2 if self.is_B_diagonal else batch_outer_product(tau, tau)
                c_tau_square = th.einsum("k,k...->k...", c, tau_square)
                B_update_inv = self.B0_inv + self.c0_tau0_2 + r_x_square + c_tau_square
                B_update = 1 / B_update_inv if self.is_B_diagonal else solve(B_update_inv, th.ones(D))

                return u_update, v_update, tau_update, c_update, n_update, B_update

            @staticmethod
            def forward(ctx, x, u, v, tau, c, n, B):
                r, elbo = GaussianDPMMFunction.__dpmm_computation__( x, u, v, tau, c, n, B)
                ctx.save_for_backward(x, r, u, v, tau, c, n, B)
                return r, elbo

            @staticmethod
            @once_differentiable
            def backward(ctx, pi_grad, elbo_grad):
                x, r, *var_params = ctx.saved_tensors
                var_params_update = GaussianDPMMFunction.__dpmm_update__(x, r, *var_params)

                # The natural gradient is the difference between the current value and the new one
                # We also consider elbo_grad to mimic the backpropagation. It should be always 1.
                var_params_grad = tuple((var_params[i] - var_params_update[i])*elbo_grad for i in range(len(var_params)))

                # there is no gradient for the data x (for now)
                x_grad = None
                return (x_grad,) + var_params_grad

        return GaussianDPMMFunction


if __name__ == "__main__":
    from .__bnpy_test__ import *
    BS, K, D = 100, 20, 4


    # WE ASSUME B0 is always diagonal
    def test_fun(x, u, v, alphaDP, tau, c, n, B, tau0, c0, n0, B0, is_diagonal):
        m_type = "diag" if is_diagonal else "full"

        B_inv = th.diag_embed(1/B) if is_diagonal else th.linalg.inv(B)
        B0_inv = th.diag_embed(1 / B0)

        bnpy_hmodel = get_bnpy_impl(K, D, u, v, alphaDP, tau, c, n, B_inv, tau0, c0, n0, B0_inv)

        # test E_log_stick
        my_val = E_log_pi(u, v)
        bnpy_val = get_bnpy_E_log_pi(bnpy_hmodel)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_pi is not correct"

        # test E_log_prec
        my_val = E_log_prec(n, B, is_diagonal=is_diagonal)
        bnpy_val = get_bnpy_E_log_prec(bnpy_hmodel)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_perc is not correct"

        # test E_log_x
        my_val = E_log_x(x, tau, c, n, B, is_diagonal=is_diagonal)
        bnpy_val = get_bnpy_E_log_x(bnpy_hmodel, x)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_x is not correct"

        # test forward function
        f = GaussianDPMMFunctionGenerator(alphaDP,
                                          tau0.view(1, -1).expand(K, -1), c0.view(1).expand(K),
                                          n0.view(1).expand(K), B0.view(1, -1).expand(K, -1),
                                          is_B_diagonal=is_diagonal, is_B0_diagonal=True).get_function()
        my_pi, my_elbo = f.__dpmm_computation__(x, u, v, tau, c, n, B)
        bnpy_pi, bnpy_elbo, *bnpy_updates = get_bnpy_train_it_results(bnpy_hmodel, x)

        # test pi and elbo
        assert th.all(th.isclose(my_pi, th.tensor(bnpy_pi).float())), f"{m_type}: pi is not correct"
        assert th.all(th.isclose(my_elbo, th.tensor(bnpy_elbo).float())), f"{m_type}: elbo is not correct"

        # test the update
        my_updates = f.__dpmm_update__(x, my_pi, u, v, tau, c, n, B)
        params_name = ['u', 'v', 'tau', 'c', 'n']
        for i in range(len(params_name)):
            my_val = my_updates[i]
            bnpy_val = bnpy_updates[i]
            name = params_name[i]

            assert th.all(
                th.isclose(my_val, th.tensor(bnpy_val).float(), atol=1e-3)), f"{m_type}: update {name} is not correct"

    ####################################################################################################################
    # diagonal with easy params
    alphaDP = th.tensor(2).float()
    tau0 = th.zeros(D).float()
    c0 = th.tensor(1).float()
    n0 = th.tensor(D).float()
    B0 = th.ones(D)

    u = th.ones(K)
    v = th.ones(K)
    tau = th.zeros(K, D)
    c = th.ones(K)
    n = D * th.ones(K)
    B = th.ones(K, D)

    x = th.rand(BS, D)

    test_fun(x, u, v, alphaDP, tau, c, n, B, tau0, c0, n0, B0, is_diagonal=True)


    # diagonal with random params
    alphaDP = th.tensor(4).float()
    tau0 = th.randn(D).float()
    c0 = 2 + th.rand(size=[1])
    n0 = D + th.randint(D, size=[1]).float()
    B0 = th.randn(D) ** 2

    u = th.rand(K)
    v = th.rand(K)
    tau = th.randn(K, D)
    c = 2 + th.rand(K)
    n = D + th.randint(D, size=[K]).float()
    B = 2 + th.randn(K, D)**2

    x = th.rand(BS, D)

    test_fun(x, u, v, alphaDP, tau, c, n, B, tau0, c0, n0, B0, is_diagonal=True)


    ####################################################################################################################
    # full
    # TODO: implement test on full cov mat
    n = D + th.randint(D, size=[K]).float()
    b = th.randn(K, D, D)
    tau = th.randn(K, D)
    c = 2 + th.rand(K)
    x = th.rand(BS, D)
    B = 5 * th.diag_embed(th.ones(K, D)) + th.einsum('bij,bkj->bik', b, b)


