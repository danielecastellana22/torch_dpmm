import torch as th
from .base import DPMM
from torch_dpmm.bayesian_distributions import FullNormalINIW, DiagonalNormalNIW, SingleNormalNIW, UnitNormalSpherical
from sklearn.cluster import kmeans_plusplus

__all__ = ['FullGaussianDPMM', 'DiagonalGaussianDPMM', 'UnitGaussinaDPMM', 'SingleGaussianDPMM']


def _get_kmeansplusplus_init(x, K):
    x_np = x.detach().numpy()
    # initialisation makes the difference: we should cover the input space
    mean_np, _ = kmeans_plusplus(x_np, K)
    return th.tensor(mean_np, device=x.device)


class FullGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(FullGaussianDPMM, self).__init__(K, D, alphaDP, FullNormalINIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None):
        c = th.ones(self.K)
        n = (self.D+2) * th.ones(self.K)

        # var params of emission
        if x is not None:
            tau = _get_kmeansplusplus_init(x, self.K)
            var = th.var(x)
        else:
            tau = th.zeros(self.K, self.D)
            var = 1

        B = th.diag_embed(var * th.ones(self.K, self.D))

        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class DiagonalGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(DiagonalGaussianDPMM, self).__init__(K, D, alphaDP, DiagonalNormalNIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None):
        c = th.ones(self.K)
        n = (self.D+2) * th.ones(self.K)

        # var params of emission
        if x is not None:
            tau = _get_kmeansplusplus_init(x, self.K)
            var = th.var(x)
        else:
            tau = th.zeros(self.K, self.D)
            var = 1

        B = var * th.ones(self.K, self.D)

        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class SingleGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(SingleGaussianDPMM, self).__init__(K, D, alphaDP, SingleNormalNIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None):
        c = th.ones(self.K)
        n = 2 * th.ones(self.K)

        # var params of emission
        if x is not None:
            tau = _get_kmeansplusplus_init(x, self.K)
            var = th.var(x)
        else:
            tau = th.zeros(self.K, self.D)
            var = 1

        B = var * th.ones(self.K)

        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class UnitGaussinaDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam):
        super(UnitGaussinaDPMM, self).__init__(K, D, alphaDP, UnitNormalSpherical, [mu0, lam])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None):
        c = th.ones(self.K)

        # var params of emission
        if x is not None:
            tau = _get_kmeansplusplus_init(x, self.K)
        else:
            tau = th.zeros(self.K, self.D)

        return self.emission_distr_class.common_to_natural([tau, c])