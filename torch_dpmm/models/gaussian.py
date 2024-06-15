import torch as th
from .base import DPMM
from torch_dpmm.prob_utils.conjugate_priors import FullINIWPrior, DiagonalNIWPrior
from sklearn.cluster import kmeans_plusplus


class BaseGaussianDPMM(DPMM):

    def _get_init_val_cov_matrices(self, x: th.Tensor = None):
        raise NotImplementedError('This should be implemented in the subclasses!')

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None):
        c = th.ones(self.K)
        n = (self.D+2) * th.ones(self.K)

        # var params of emission
        if x is not None:
            x_np = x.detach().numpy()
            # initialisation makes the difference: we should cover the input space
            mean_np, _ = kmeans_plusplus(x_np, self.K)
            tau = th.tensor(mean_np)
        else:
            tau = th.zeros(self.K, self.D)

        B = self._get_init_val_cov_matrices(x)

        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class FullGaussianDPMM(BaseGaussianDPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(FullGaussianDPMM, self).__init__(K, D, alphaDP, FullINIWPrior, [mu0, lam, Phi, nu])

    def _get_init_val_cov_matrices(self, x: th.Tensor = None):
        v = 1
        if x is not None:
            v = th.var(x)

        return th.diag_embed(v * th.ones(self.K, self.D))


class DiagonalGaussianDPMM(BaseGaussianDPMM):

    def __init__(self, K, D, alphaDP, mu0, lam, Phi, nu):
        super(DiagonalGaussianDPMM, self).__init__(K, D, alphaDP, DiagonalNIWPrior, [mu0, lam, Phi, nu])

    def _get_init_val_cov_matrices(self, x: th.Tensor = None):
        v = 1
        if x is not None:
            v = th.var(x)

        return v * th.ones(self.K, self.D)