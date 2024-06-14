import torch as th
from .__base__ import ExponentialFamilyDistribution


class Beta(ExponentialFamilyDistribution):

    # x ~ Beta(alpha, beta)
    # common params: [alpha, beta]
    # natural params: [eta_1, eta_2]

    @classmethod
    def __h_x__(cls, x):
        return 1/(x[0] * (1-x[0]))

    @classmethod
    def __A_eta__(cls, eta):
        return th.lgamma(eta[0]) + th.lgamma(eta[1]) - th.lgamma(eta[0] + eta[1])

    @classmethod
    def __T_x__(cls, x, idx=None):
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


if __name__ == '__main__':
    from torch.distributions import Beta as th_Beta
    from torch.distributions import kl_divergence
    B, K, D = 100, 10, 4

    ####################################################################################################################
    # test beta_kl_div
    alpha_0, beta_0 = th.rand(K), th.rand(K)
    alpha_1, beta_1 = th.rand(K), th.rand(K)
    my_kl = Beta.kl_div([alpha_0, beta_0], [alpha_1, beta_1])
    th_kl = kl_divergence(th_Beta(alpha_0, beta_0), th_Beta(alpha_1, beta_1))
    assert th.isclose(my_kl.sum(), th_kl.sum()), "Beta_kl"
    ####################################################################################################################
