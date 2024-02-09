import numpy as np

# we take code this code from bnpy to test the correcteness of our implementation.
# https://github.com/bnpy/bnpy


from .dpModel import Mock_DP
from .hmodel import Mock_HModel
from .guassModel import Mock_Gauss
from .bags import ParamBag


class Mock_Data:
    def __init__(self, s):
        self.X = s.numpy()
        self.nObs = s.shape[0]
        self.dim = s.shape[1]


def get_bnpy_impl(K, D, u, v, alphaDP, tau, c, n, B_inv, tau0, c0, n0, B0_inv):
    post_params = ParamBag(K=K, D=D, m=tau.numpy(), kappa=c.numpy(), nu=n.numpy(), B=B_inv.numpy())
    prior_params = ParamBag(D=D, m=tau0.numpy(), kappa=c0.numpy(), nu=n0.numpy(), B=B0_inv.numpy())

    obs = Mock_Gauss(K, D, post_params, prior_params)
    alloc = Mock_DP(K, u.numpy(), v.numpy(), 1, alphaDP.numpy())

    return Mock_HModel(alloc, obs)


def get_bnpy_E_log_pi(hmodel):
    return hmodel.allocModel.Elogbeta


def get_bnpy_E_log_prec(hmodel):
    return hmodel.obsModel.Cache['E_logdetL']


def get_bnpy_E_log_x(hmodel, x):
    data = Mock_Data(x)
    return hmodel.obsModel.calcLogSoftEvMatrix_FromPost(data)

def get_bnpy_train_it_results(hmodel, x):
    data = Mock_Data(x)
    LP = hmodel.calc_local_params(data)
    pi = LP['resp'].copy()
    SS = hmodel.get_global_suff_stats(data, LP)
    elbo = hmodel.calc_evidence(data)
    hmodel.update_global_params(SS)
    return (pi, elbo,
            hmodel.allocModel.eta1, hmodel.allocModel.eta0,
            hmodel.obsModel.Post.m, hmodel.obsModel.Post.kappa, hmodel.obsModel.Post.nu, hmodel.obsModel.Post.B)