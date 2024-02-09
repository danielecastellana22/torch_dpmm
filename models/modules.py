import torch as th
import torch.nn as nn
import torch.nn.init as INIT
from sklearn.cluster import kmeans_plusplus
from prob_utils.constraints import *
from .functions import GaussianDPMMFunctionGenerator


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

        self.is_B0_diagonal = not (B0.ndim >= 2 and B0.shape[-2] == (D, D))
        if self.is_B0_diagonal:
            self.B0 = self.__validate_arg__("B0", B0, (K, D), [Positive()])
        else:
            # we enforce diagonality on B0. If it is full, we can have numerical error due to inverse
            raise NotImplementedError('The current implementation can lead to numerical errors if the prior '
                                      'of the precision is parametrised by a full matrix!')
            self.B0 = self.__validate_arg__("B0", B0, (K, D, D), [PositiveDefinite()])

        # get the Function associated to this prior params
        self.__dpmm_func__ = GaussianDPMMFunctionGenerator(self.alphaDP, self.tau0, self.c0, self.n0, self.B0,
                                                           self.is_diagonal, self.is_B0_diagonal).get_function().apply

        # define the variational parameters
        # var params of beta (stick-breaking)
        self.u = th.nn.Parameter(th.empty(K))
        self.v = th.nn.Parameter(th.empty(K))

        # var params of emission
        self.tau = th.nn.Parameter(th.empty(K, D))  # mean of mu posterior
        self.c = th.nn.Parameter(th.empty(K))  # precision coeff. of mu posterior

        self.n = th.nn.Parameter(th.empty(K))  # deg_of_freedom of precision posterior wishart(n,B)
        self.B = th.nn.Parameter(th.empty(K, D) if is_diagonal else th.empty(K, D, D))  # B matrix of precision posterior wishart(n,B)
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
        # var params of beta (stick-breaking)
        INIT.constant_(self.u, 1)
        INIT.constant_(self.v, 1)

        # var params of emission
        if x is not None:
            x_np = x.detach().numpy()
            # initialisation makes the difference: we should cover the input space
            mean_np, _ = kmeans_plusplus(x_np, self.K)
            self.tau.data = th.tensor(mean_np)
        else:
            INIT.zeros_(self.tau)

        INIT.constant_(self.c, 1)

        INIT.constant_(self.n, self.D)

        eye_val = th.ones(self.K, self.D)
        self.B.data = eye_val if self.is_diagonal else th.diag_embed(eye_val)

    def forward(self, x):
        pi, elbo = self.__dpmm_func__(x, self.u, self.v, self.tau, self.c,
                                      self.n, self.B)
        return pi, elbo

    def get_component_weights(self):
        with th.no_grad():
            expected_beta = self.u / (self.u + self.v)
            beta1m_cumprod = (1 - expected_beta).cumprod(-1)
            return F.pad(expected_beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def n_used_components(self):
        return th.sum(self.get_component_weights() > 1e-2).item()


if __name__ == '__main__':
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    th.manual_seed(1234)
    import torch.optim as optim
    import numpy as np
    from torch.distributions import MultivariateNormal
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from .__bnpy_test__ import get_bnpy_impl, get_bnpy_train_it_results

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
    lr = 1
    num_iterations = 500

    # prior params
    alphaDP = th.tensor(1).float()
    tau0 = th.zeros(D).float()
    c0 = th.tensor(1).float()
    n0 = th.tensor(D).float()
    B0 = th.ones(D).float()

    my_DPMM = GaussianDPMM(K=K, D=D, alphaDP=alphaDP, tau0=tau0, c0=c0, n0=n0, B0=B0, is_diagonal=True)

    param_names_to_update = []
    param_names_to_update.append('u')
    param_names_to_update.append('v')
    param_names_to_update.append('tau')
    param_names_to_update.append('c')
    param_names_to_update.append('n')
    param_names_to_update.append('B')

    my_optim = optim.SGD(params=[my_DPMM.get_parameter(n) for n in param_names_to_update], lr=lr)

    elbo_list = []
    pbar = tqdm(range(num_iterations))
    for j in pbar:

        my_optim.zero_grad()
        pi, elbo = my_DPMM(x)
        elbo_list.append(elbo.detach().item())

        # get bnpy val
        bnpy_hmodel = get_bnpy_impl(K, D,
                                    my_DPMM.u.detach(), my_DPMM.v.detach(), alphaDP,
                                    my_DPMM.tau.detach(), my_DPMM.c.detach(),
                                    my_DPMM.n.detach(), th.diag_embed(1/my_DPMM.B.detach()),
                                    tau0, c0, n0, th.diag_embed(1/B0))

        bnpy_pi, bnpy_elbo, *bnpy_updates = get_bnpy_train_it_results(bnpy_hmodel, x)

        # test pi and elbo
        assert th.all(th.isclose(pi.detach(), th.tensor(bnpy_pi).float(), atol=1e-3)), "pi diag is not correct"
        assert th.all(th.isclose(elbo.detach(), th.tensor(bnpy_elbo).float())), "elbo diag is not correct"

        pbar.set_postfix({'ELBO loss': elbo.detach().item()})

        elbo.backward()
        my_optim.step()

        param_names = ['u', 'v', 'tau', 'c', 'n']
        for i in range(len(param_names)):
            name = param_names[i]
            if name in param_names_to_update:
                my_val = my_DPMM.get_parameter(name).detach()
                bnpy_val = bnpy_updates[i]
                if lr == 1:
                    assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float(), atol=1e-3)), f"update {name} is not correct"

    a = 2
    elbo_array = np.array(elbo_list)
    plt.figure()
    plt.plot(elbo_array)
    plt.title('Full ELBO')
    plt.show()

    #assert np.all((elbo_array[1:] - elbo_array[:-1]) > -1e-2)