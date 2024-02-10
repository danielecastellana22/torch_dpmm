import numpy as np

# we take code this code from bnpy to test the correcteness of our implementation.
# https://github.com/bnpy/bnpy


from .dpModel import Mock_DP
from .hmodel import Mock_HModel
from .guassModel import Mock_Gauss
from .bags import ParamBag


class Mock_Data:
    def __init__(self, s):
        self.X = s.numpy().copy()
        self.nObs = s.shape[0]
        self.dim = s.shape[1]


def get_bnpy_impl(K, D, u, v, alphaDP, tau, c, n, B_inv, tau0, c0, n0, B0_inv):
    post_params = ParamBag(K=K, D=D, m=tau.numpy().copy(), kappa=c.numpy().copy(),
                           nu=n.numpy().copy(), B=B_inv.numpy().copy())
    prior_params = ParamBag(D=D, m=tau0.numpy().copy(), kappa=c0.numpy().copy(),
                            nu=n0.numpy().copy(), B=B0_inv.numpy().copy())

    obs = Mock_Gauss(K, D, post_params, prior_params)
    alloc = Mock_DP(K, u.numpy().copy(), v.numpy().copy(), 1, alphaDP.numpy().copy())

    return Mock_HModel(alloc, obs)


def get_bnpy_E_log_pi(hmodel):
    return hmodel.allocModel.Elogbeta


def get_bnpy_E_log_prec(hmodel):
    return hmodel.obsModel.Cache['E_logdetL']


def get_bnpy_E_log_x(hmodel, x):
    data = Mock_Data(x)
    return hmodel.obsModel.calcLogSoftEvMatrix_FromPost(data)


def get_bnpy_common_to_natural(hmodel):
    hmodel.obsModel.convertPostToNatural()

    return (hmodel.obsModel.Post.km, hmodel.obsModel.Post.kappa,
            hmodel.obsModel.Post.nu, hmodel.obsModel.Post.Bnat)


def get_bnpy_natural_to_common(hmodel):
    K = hmodel.obsModel.Post.K
    D = hmodel.obsModel.Post.D
    old_param = hmodel.obsModel.Post
    hmodel.obsModel.Post = ParamBag(K=K, D=D, km=old_param.m, kappa=old_param.kappa,
                                    nu=old_param.nu, Bnat=old_param.B)
    hmodel.obsModel.convertPostToCommon()

    return (hmodel.obsModel.Post.m, hmodel.obsModel.Post.kappa,
            hmodel.obsModel.Post.nu, hmodel.obsModel.Post.B)


def get_bnpy_train_it_results(hmodel, x, lr):
    data = Mock_Data(x)
    LP = hmodel.calc_local_params(data)
    pi = LP['resp'].copy()
    SS = hmodel.get_global_suff_stats(data, LP)
    elbo = hmodel.calc_evidence(data)
    hmodel.obsModel.convertPostToNatural()
    prev_nat_values = (hmodel.allocModel.eta1.copy(), hmodel.allocModel.eta0.copy(),
                       hmodel.obsModel.Post.km.copy(), hmodel.obsModel.Post.kappa.copy(),
                       hmodel.obsModel.Post.nu.copy(), hmodel.obsModel.Post.Bnat.copy())
    
    new_nu, new_Bnat, new_km, new_kappa = hmodel.obsModel.calcNaturalPostParams(SS)
    emission_nat_updates = (new_km, new_kappa, new_nu, new_Bnat)
    hmodel.update_global_params(SS, rho=lr)
    
    new_nat_values = (hmodel.allocModel.eta1.copy(), hmodel.allocModel.eta0.copy(),
                       hmodel.obsModel.Post.km.copy(), hmodel.obsModel.Post.kappa.copy(),
                       hmodel.obsModel.Post.nu.copy(), hmodel.obsModel.Post.Bnat.copy())
    
    new_comm_values = (hmodel.allocModel.eta1.copy(), hmodel.allocModel.eta0.copy(),
                       hmodel.obsModel.Post.m.copy(), hmodel.obsModel.Post.kappa.copy(),
                       hmodel.obsModel.Post.nu.copy(), hmodel.obsModel.Post.B.copy())
    
    return (pi, elbo, new_comm_values, prev_nat_values, new_nat_values, emission_nat_updates)
