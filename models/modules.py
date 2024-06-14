import torch as th
import torch.nn as nn
from sklearn.cluster import kmeans_plusplus
from torch.distributions import MultivariateNormal
from ..prob_utils.constraints import *
from ..prob_utils.misc import log_normalise
from ..prob_utils.conjugate_priors import ConjugatePriorDistribution, StickBreakingPrior


class DPMM(nn.Module):

    def __init__(self, K, D, mix_weights_prior_common_params, emission_conj_prior_class: ConjugatePriorDistribution,
                 emission_conj_prior_common_params: list[th.Tensor]):
        super().__init__()

        self.K = K  # number of mixture components
        self.D = D  # size of output vector

        # store the prior nat params of the mixture weights
        mix_weights_prior_common_params = StickBreakingPrior.validate_common_params(mix_weights_prior_common_params)
        for i, p in enumerate(StickBreakingPrior.common_to_natural(mix_weights_prior_common_params)):
            self.register_buffer(f'mix_prior_eta_{i}', p)
            self.register_parameter(f'mix_var_eta_{i}', nn.Parameter(th.empty_like(p)))

        emission_conj_prior_common_params = emission_conj_prior_class.validate_common_params(emission_conj_prior_common_params)
        for i, p in enumerate(emission_conj_prior_class.common_to_natural(emission_conj_prior_common_params)):
            self.register_buffer(f'emission_prior_eta_{i}', p)
            self.register_parameter(f'emission_var_eta_{i}', nn.Parameter(th.empty_like(p)))

        # get the Function associated to this prior params
        self.__update_computation_fucntion__()

        self.init_var_params()

    def __update_computation_fucntion__(self):
        self.__dpmm_func__ = GaussianDPMMFunctionGenerator(self.u0, self.v0, self.tau0, self.c0, self.n0, self.B0,
                                                           self.is_diagonal, self.is_B0_diagonal).get_function().apply



    def init_var_params(self, x=None):

        # TODO: initialisation is crucial and should be random
        u = th.ones_like(self.nat_u)
        v = th.ones_like(self.nat_u)
        c = th.ones_like(self.nat_c)
        n = (self.D+2) * th.ones_like(self.nat_n)

        # var params of emission
        if x is not None:
            x_np = x.detach().numpy()
            # initialisation makes the difference: we should cover the input space
            mean_np, _ = kmeans_plusplus(x_np, self.K)
            tau = th.tensor(mean_np)
            B_eye_val = th.var(x) * th.ones(self.K, self.D)
        else:
            tau = th.zeros_like(self.nat_tau)
            B_eye_val = th.ones(self.K, self.D)

        B = B_eye_val if self.is_diagonal else th.diag_embed(B_eye_val)

        self.__set_var_params__(u, v, tau, c, n, B)

    def __set_var_params__(self, u=None, v=None, tau=None, c=None, n=None, B=None):
        nat_u, nat_v, nat_tau, nat_c, nat_n, nat_B = common_to_natural(u, v, tau, c, n, B, self.is_diagonal)

        self.nat_u.data = nat_u
        self.nat_v.data = nat_v
        self.nat_tau.data = nat_tau
        self.nat_c.data = nat_c
        self.nat_n.data = nat_n
        self.nat_B.data = nat_B

    def forward(self, x):
        r, elbo_loss = self.__dpmm_func__(x, self.nat_u, self.nat_v, self.nat_tau, self.nat_c,
                                      self.nat_n, self.nat_B)
        return r, elbo_loss

    @th.no_grad()
    def get_var_params(self):
        params = natural_to_common(self.nat_u, self.nat_v, self.nat_tau, self.nat_c, self.nat_n, self.nat_B,
                                   self.is_diagonal)

        return (p.detach() for p in params)

    def get_num_active_components(self):
        u, v, tau, c, n, B = natural_to_common(self.nat_u, self.nat_v,
                                               self.nat_tau, self.nat_c, self.nat_n, self.nat_B, self.is_diagonal)

        sticks = u / (u + v)
        log_1_minus_sticks = th.log(1 - sticks)
        r = th.exp(th.cumsum(log_1_minus_sticks, -1) - log_1_minus_sticks + th.log(sticks))
        return th.sum(r > 0.01).item()

    @th.no_grad()
    def get_expected_params(self, return_log_r=False):
        u, v, tau, c, n, B = natural_to_common(self.nat_u, self.nat_v,
                                               self.nat_tau, self.nat_c, self.nat_n, self.nat_B, self.is_diagonal)

        sticks = u / (u + v)
        log_1_minus_sticks = th.log(1 - sticks)
        log_r = th.cumsum(log_1_minus_sticks, -1) - log_1_minus_sticks + th.log(sticks)
        mu = tau
        sigma = (B if not self.is_diagonal else th.diag_embed(B)) / (n - self.D - 1).view(-1, 1, 1)

        if return_log_r:
            return log_r, mu, sigma
        else:
            return log_r.exp(), mu, sigma

    @th.no_grad()
    def get_expected_log_likelihood(self, x):
        log_r, mu, sigma = self.get_expected_params(return_log_r=True)
        # TODO: implement our computation instead of relying on pytorch.
        #  SIGMA CAN BE NOT POSITIVE DEFINITE DUE TO NUMERICAL ERROR
        sigma = sigma + th.diag_embed(1e-3 * th.ones(sigma.shape[0], self.D))
        exp_loglike = MultivariateNormal(loc=mu, covariance_matrix=sigma).log_prob(x.unsqueeze(1))
        _, logZ = log_normalise(exp_loglike+log_r)
        return logZ


if __name__ == '__main__':
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    th.manual_seed(1234)
    import torch.optim as optim
    import numpy as np
    from torch.distributions import MultivariateNormal
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    num_classes = 4
    D = 2

    x = th.cat((MultivariateNormal(-8 * th.ones(D), th.eye(D)).sample([50]),
                MultivariateNormal(8 * th.ones(D), th.eye(D)).sample([50]),
                MultivariateNormal(th.tensor([1.5, 2]), th.eye(D)).sample([50]),
                MultivariateNormal(th.tensor([-0.5, 1]), th.eye(D)).sample([50])))

    y = th.cat((0 * th.ones(50),
                1 * th.ones(50),
                2 * th.ones(50),
                3 * th.ones(50))).to(th.int64)

    K = 20
    D = 2

    num_iterations = 500

    # prior params
    alphaDP = th.tensor(1).float()
    tau0 = th.zeros(D).float()
    c0 = th.tensor(1).float()
    n0 = th.tensor(D).float()
    B0 = th.ones(D).float()

    my_DPMM = GaussianDPMM(K=K, D=D, alphaDP=alphaDP, tau0=tau0, c0=c0, n0=n0, B0=B0, is_diagonal=True)

    lr = 0.01
    param_names_to_update = []
    param_names_to_update.append('nat_u')
    param_names_to_update.append('nat_v')
    param_names_to_update.append('nat_tau')
    param_names_to_update.append('nat_c')
    param_names_to_update.append('nat_n')
    param_names_to_update.append('nat_B')

    my_optim = optim.SGD(params=[my_DPMM.get_parameter(n) for n in param_names_to_update], lr=lr)

    elbo_list = []
    pbar = tqdm(range(num_iterations))
    for j in pbar:

        my_optim.zero_grad()
        pi, elbo = my_DPMM(x)
        pbar.set_postfix({'ELBO loss': elbo.detach().item()})
        elbo_list.append(elbo.detach().item())
        elbo.backward()
        my_optim.step()

    a = 2
    elbo_array = np.array(elbo_list)
    plt.figure()
    plt.plot(elbo_array)
    plt.title('Full ELBO')
    plt.show()

    #assert np.all((elbo_array[1:] - elbo_array[:-1]) > -1e-2)