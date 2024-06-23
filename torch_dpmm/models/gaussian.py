import torch as th
from .base import DPMM
from torch_dpmm.bayesian_distributions import FullNormalINIW, DiagonalNormalNIW, SingleNormalNIW, UnitNormalSpherical
from sklearn.cluster import kmeans_plusplus

__all__ = ['FullGaussianDPMM', 'DiagonalGaussianDPMM', 'UnitGaussinaDPMM', 'SingleGaussianDPMM']


def _get_gaussian_init_vals(x, D, mask, v_c=1, v_n=1):
    K = th.sum(mask).item()

    # compute initialisation for tau
    if x is None:
        tau =  th.zeros([K, D], device=mask.device)
    else:
        x_np = x.detach().numpy()
        # initialisation makes the difference: we should cover the input space
        if x_np.shape[0] > 3 * K:
            # there are enough sample to init all K clusters
            mean_np, _ = kmeans_plusplus(x_np, K)
            tau = th.tensor(mean_np, device=mask.device)
        else:
            # there are few samples
            to_init = int(x_np.shape[0] / 3)
            mean_np, _ = kmeans_plusplus(x_np, to_init)
            kmeans_init_mask = th.logical_and(mask, th.cumsum(mask, dim=0) <= to_init)
            tau = th.zeros([K, D], device=mask.device)
            tau[kmeans_init_mask] = th.tensor(mean_np, device=mask.device)

    # compute initialisation for B
    B = th.tensor(1.0, device=mask.device)
    if x is not None:
        B = th.var(x) * B

    # compute initialisation for c
    c = v_c * th.ones([K], device=mask.device)

    # compute initialisation for n
    n = v_n * th.ones([K], device=mask.device)

    return tau, c, B, n


class FullGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(FullGaussianDPMM, self).__init__(K, D, alphaDP, FullNormalINIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask, v_c=1, v_n=self.D+2)
        B = th.diag_embed(B*th.ones_like(tau))
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class DiagonalGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(DiagonalGaussianDPMM, self).__init__(K, D, alphaDP, DiagonalNormalNIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None, mask=None):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask, v_c=1, v_n=self.D + 2)
        B = B*th.ones_like(tau)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class SingleGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(SingleGaussianDPMM, self).__init__(K, D, alphaDP, SingleNormalNIW, [mu0, lam, Phi, nu])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask, v_c=1, v_n=2)
        B = B * th.ones_like(c)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class UnitGaussinaDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu0, lam):
        super(UnitGaussinaDPMM, self).__init__(K, D, alphaDP, UnitNormalSpherical, [mu0, lam])

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, _, _ = _get_gaussian_init_vals(x, self.D, mask, v_c=1)
        return self.emission_distr_class.common_to_natural([tau, c])