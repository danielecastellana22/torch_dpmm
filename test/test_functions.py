import torch as th
from test.__bnpy_test__ import *
from models.functions import *

th.manual_seed(1234)


def __create_data_and_bnpy_model__(is_diagonal):
    BS, K, D = 100, 20, 4

    # data
    x = th.rand(BS, D)

    # priors values
    alphaDP = th.tensor(4).float()
    tau0 = th.randn(D).float()
    c0 = 2 + th.rand(size=[1])
    n0 = D + th.randint(D, size=[1]).float()
    # B0 is always diagonal
    B0 = th.randn(D) ** 2
    B0_bnpy = th.diag_embed(B0)

    # var params of the stick
    u = th.rand(K)
    v = th.rand(K)

    # var params of mu
    tau = th.randn(K, D)
    c = 2 + th.rand(K)

    # var params of Lam
    n = D + th.randint(D, size=[K]).float()

    if is_diagonal:
        # diagonal with random var params
        B = 5 + th.randn(K, D) ** 2
        B_bnpy = th.diag_embed(B)
    else:
        b = th.randn(K, D, D)
        B = 5 * th.diag_embed(th.ones(K, D)) + th.einsum('bij,bkj->bik', b, b)
        B_bnpy = B

    bnpy_hmodel = get_bnpy_impl(K, D, u, v, alphaDP, tau, c, n, B_bnpy, tau0, c0, n0, B0_bnpy)
    return bnpy_hmodel, x, (u, v), (tau, c, n, B), (alphaDP,), (tau0, c0, n0, B0)


def test_e_log_pi():
    # it does not depend on diagonal/full
    bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
        is_diagonal=True)
    my_val = E_log_pi(*stick_post)
    bnpy_val = get_bnpy_E_log_pi(bnpy_hmodel)
    assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"E_log_pi is not correct"


def test_e_log_prec():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
            is_diagonal=is_diagonal)
        n, B = emisssion_post[-2], emisssion_post[-1]
        B_inv = 1/B if is_diagonal else th.linalg.inv(B)

        my_val = E_log_prec(n, B_inv, is_diagonal=is_diagonal)
        bnpy_val = get_bnpy_E_log_prec(bnpy_hmodel)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_perc is not correct"


def test_e_log_x():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
            is_diagonal=is_diagonal)
        tau, c, n, B = emisssion_post
        B_inv = 1/B if is_diagonal else th.linalg.inv(B)
        my_val = E_log_x(x, tau, c, n, B_inv, is_diagonal=is_diagonal)
        bnpy_val = get_bnpy_E_log_x(bnpy_hmodel, x)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_x is not correct"


def test_natural_to_common():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
            is_diagonal=is_diagonal)

        _, _, *my_common_val_emission = natural_to_common(*(stick_post + emisssion_post), is_diagonal=is_diagonal)
        bnpy_common_val_emission = get_bnpy_natural_to_common(bnpy_hmodel)
        params_name = ['tau', 'c', 'n', 'B']
        for i in range(len(params_name)):
            name = params_name[i]
            my_val = my_common_val_emission[i]
            bnpy_val = bnpy_common_val_emission[i]
            if name == 'B' and is_diagonal:
                bnpy_val = np.diagonal(bnpy_val, axis1=-2, axis2=-1)

            assert th.all(th.isclose(my_val[i], th.tensor(
                bnpy_val[i]).float())), f"{m_type}: common_to_natural on {name} is not correct"


def test_common_to_natural():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
            is_diagonal=is_diagonal)

        _, _, *my_nat_val_emission = common_to_natural(*(stick_post + emisssion_post), is_diagonal=is_diagonal)
        bnpy_nat_val_emission = get_bnpy_common_to_natural(bnpy_hmodel)
        params_name = ['tau', 'c', 'n', 'B']
        for i in range(len(params_name)):
            name = params_name[i]
            my_val = my_nat_val_emission[i]
            bnpy_val = bnpy_nat_val_emission[i]
            if name == 'B' and is_diagonal:
                bnpy_val = np.diagonal(bnpy_val, axis1=-2, axis2=-1)

            assert th.all(th.isclose(my_val[i], th.tensor(
                bnpy_val[i]).float())), f"{m_type}: common_to_natural on {name} is not correct"


def test_gaussian_dpmmfunction_generator():

    class FakeCtx:
        def __init__(self):
            self.saved_tensors = []

        def save_for_backward(self, *args):
            self.saved_tensors = args

    lr=0.01

    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, comm_emisssion_post, stick_prior, emission_prior = __create_data_and_bnpy_model__(
            is_diagonal=is_diagonal)

        alphaDP, = stick_prior
        tau0, c0, n0, B0 = emission_prior
        K, D = comm_emisssion_post[0].shape

        my_nat_params = common_to_natural(*(stick_post+comm_emisssion_post), is_diagonal)

        f = GaussianDPMMFunctionGenerator(alphaDP,
                                          tau0.view(1, -1).expand(K, -1), c0.view(1).expand(K),
                                          n0.view(1).expand(K), B0.view(1, -1).expand(K, -1),
                                          is_B_diagonal=is_diagonal, is_B0_diagonal=True).get_function()

        ctx = FakeCtx()
        my_pi, my_elbo = f.forward(ctx, x, *my_nat_params)
        bnpy_pi, bnpy_elbo, _, bnpy_prev_nat_params, bnpy_new_nat_params, bnpy_nat_updates = get_bnpy_train_it_results(bnpy_hmodel, x, lr)

        # test pi and elbo
        assert th.all(th.isclose(my_pi, th.tensor(bnpy_pi).float(), atol=1e-3)), f"{m_type}: pi is not correct"
        assert th.all(th.isclose(my_elbo, th.tensor(bnpy_elbo).float())), f"{m_type}: elbo is not correct"

        # test the update
        my_nat_grads = f.backward(ctx, None, th.tensor(1.0))[1:]
        my_new_nat_params = [my_nat_params[i] - lr*my_nat_grads[i] for i in range(len(my_nat_params))]
        my_nat_updates = [my_nat_params[i] - my_nat_grads[i] for i in range(len(my_nat_grads))]
        params_name = ['u', 'v', 'tau', 'c', 'n', 'B']
        for i in range(len(params_name)):
            name = params_name[i]

            my_prev_val = my_nat_params[i]
            my_new_val = my_new_nat_params[i]

            bnpy_prev_val = bnpy_prev_nat_params[i]
            bnpy_new_val = bnpy_new_nat_params[i]
            if name == 'B' and is_diagonal:
                bnpy_prev_val = np.diagonal(bnpy_prev_val, axis1=-2, axis2=-1)
                bnpy_new_val = np.diagonal(bnpy_new_val, axis1=-2, axis2=-1)

            assert th.all(th.isclose(my_prev_val, th.tensor(bnpy_prev_val).float(), atol=1e-3))

            if i >= 2:
                my_up_val = my_nat_updates[i]
                bnpy_up_val = bnpy_nat_updates[i-2]
                if name == 'B' and is_diagonal:
                    bnpy_up_val = np.diagonal(bnpy_up_val, axis1=-2, axis2=-1)
                assert th.all(th.isclose(my_up_val, th.tensor(bnpy_up_val).float(), atol=1e-3))

            assert th.all(th.isclose(my_new_val, th.tensor(bnpy_new_val).float(), atol=1e-3))

