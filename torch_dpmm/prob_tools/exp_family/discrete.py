import torch as th
from .base import ExponentialFamilyDistribution


class Beta(ExponentialFamilyDistribution):

    # x ~ Beta(alpha, beta)
    # common params: [alpha, beta]
    # natural params: [eta_1, eta_2]

    _theta_names = ['aplha', 'beta']
    _theta_shape_list = ['[K]', '[K]']
    _theta_constraints_list = ['Positive()', 'Positive()']

    @classmethod
    def _h_x(cls, x):
        return 1/(x[0] * (1-x[0]))

    @classmethod
    def _A_eta(cls, eta):
        return th.lgamma(eta[0]) + th.lgamma(eta[1]) - th.lgamma(eta[0] + eta[1])

    @classmethod
    def _T_x(cls, x, idx=None):
        if idx == 0:
            return th.log(x[0])
        elif idx == 1:
            return th.log(1 - x[0])
        else:
            return [th.log(x[0]), th.log(1-x[0])]

    @classmethod
    def expected_T_x(cls, eta, idx=None):
        aux = th.digamma(eta[0] + eta[1])
        if idx == 0:
            return th.digamma(eta[0]) - aux
        elif idx == 1:
            return th.digamma(eta[1]) - aux
        else:
            return [th.digamma(eta[0]) - aux, th.digamma(eta[1]) - aux]

    @classmethod
    def natural_to_common(cls, eta):
        return eta

    @classmethod
    def common_to_natural(cls, theta):
        return theta
