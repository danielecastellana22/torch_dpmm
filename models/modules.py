import torch as th
import torch.nn as nn
import torch.nn.init as INIT
from sklearn.cluster import kmeans_plusplus
from ..prob_utils.constraints import *
from .functions import GaussianDPMMFunctionGenerator, natural_to_common, common_to_natural, E_log_x
from torch.distributions import MultivariateNormal


class GaussianDPMM(nn.Module):

    # TODO: implement tied version
    def __init__(self, K, D, alphaDP, tau0, c0, n0, B0, is_diagonal):
        super().__init__()

        self.K = K  # number of mixture components
        self.D = D  # size of output vector
        self.is_diagonal = is_diagonal  # the posterior of the precision is diagonal

        # store the prior args
        self.alphaDP = self.__validate_arg__("alphaDP", alphaDP, tuple(), [Positive()])
        self.tau0 = self.__validate_arg__("tau0", tau0, (K, D), [])
        self.c0 = self.__validate_arg__("c0", c0, (K,), [Positive()])
        self.n0 = self.__validate_arg__("n0", n0, (K,), [GreaterThan(D-1)])

        if th.is_tensor(B0) and B0.ndim > 2:
            # we enforce diagonality on B0. If it is full, we can have numerical error due to inverse
            raise NotImplementedError('The current implementation can lead to numerical errors if the prior '
                                      'of the precision is parametrised by a full matrix!')

        self.is_B0_diagonal = True
        self.B0 = self.__validate_arg__("B0", B0, (K, D), [Positive()])

        # get the Function associated to this prior params
        self.__dpmm_func__ = GaussianDPMMFunctionGenerator(self.alphaDP, self.tau0, self.c0, self.n0, self.B0,
                                                           self.is_diagonal, self.is_B0_diagonal).get_function().apply

        # define the variational parameters
        # var params of beta (stick-breaking)
        self.nat_u = th.nn.Parameter(th.empty(K))
        self.nat_v = th.nn.Parameter(th.empty(K))

        # var params of emission
        self.nat_tau = th.nn.Parameter(th.empty(K, D))  # mean of mu posterior
        self.nat_c = th.nn.Parameter(th.empty(K))  # precision coeff. of mu posterior

        self.nat_n = th.nn.Parameter(th.empty(K))  # deg_of_freedom of precision posterior wishart(n,B)
        self.nat_B = th.nn.Parameter(th.empty(K, D) if is_diagonal else th.empty(K, D, D))  # B matrix of precision posterior wishart(n,B)
        self.init_var_params()

    @staticmethod
    def __validate_arg__(name, value, expected_shape, constraints_list):

        if not isinstance(value, th.Tensor):
            value = th.tensor(value).float()

        if value.ndim == 0:
            value = value.view((1,) * len(expected_shape)).expand(expected_shape)

        # check the shape
        if value.ndim > len(expected_shape):
            raise ValueError(f'{name} has too many dimensions: we got {value.shape} but we expected {expected_shape}!')
        elif value.ndim == len(expected_shape)-1:
            value = value.unsqueeze(0).expand(expected_shape[:1] + tuple([-1]*(len(expected_shape)-1)))
        elif value.ndim < len(expected_shape)-1:
            raise ValueError(f'{name} has too few dimensions! We broadcast only along the first dimension:'
                             f' we got {value.shape} but we expected {expected_shape}!')

        assert value.ndim == len(expected_shape)
        if value.shape != expected_shape:
            raise ValueError(f'{name} has the wrong shape: we got {value.shape} but we expected {expected_shape}!')

        # check the constraints
        for c in constraints_list:
            if not c(value):
                raise ValueError(c.message(name, 'Gaussian-DPMM'))

        return value

    def init_var_params(self, x=None):

        # TODO: initialisation is crucial
        # these params are the same in the natural and in the common form
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
    def get_expected_params(self):
        u, v, tau, c, n, B = natural_to_common(self.nat_u, self.nat_v,
                                               self.nat_tau, self.nat_c, self.nat_n, self.nat_B, self.is_diagonal)

        sticks = u / (u + v)
        log_1_minus_sticks = th.log(1 - sticks)
        r = th.exp(th.cumsum(log_1_minus_sticks, -1) - log_1_minus_sticks + th.log(sticks))

        mu = tau
        sigma = (B if not self.is_diagonal else th.diag_embed(B)) / (n - self.D - 1).view(-1, 1, 1)

        # TODO: how to select the reshold? 1/100*alph as pyro?
        mask = r > 1e-3
        return r[mask], mu[mask], sigma[mask]

    @th.no_grad()
    def get_expected_log_likelihood(self, x):
        r, mu, sigma = self.get_expected_params()
        # TODO: implement your computation instead of relying on pytorch
        exp_loglike = MultivariateNormal(loc=mu, covariance_matrix=sigma).log_prob(x.unsqueeze(1))
        # TODO: maybe the max should go outside? what about the r?
        return th.max(exp_loglike,  -1)[0]


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