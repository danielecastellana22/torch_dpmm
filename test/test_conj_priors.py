import torch as th
from .utils.bnpy_test import *
from .utils.misc import *
from torch_dpmm.prob_utils.conjugate_priors import StickBreakingPrior, DiagonalNIWPrior, FullINIWPrior
th.manual_seed(1234)


def test_e_log_pi():
    # it does not depend on diagonal/full
    bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = create_data_and_bnpy_model(
        is_diagonal=True)
    my_val = StickBreakingPrior.expected_log_params(StickBreakingPrior.common_to_natural(list(stick_post)))[0]
    bnpy_val = get_bnpy_E_log_pi(bnpy_hmodel)
    assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"E_log_pi is not correct"


def test_common_natural_conversions():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = create_data_and_bnpy_model(
            is_diagonal=is_diagonal)
        conj_distr = DiagonalNIWPrior if is_diagonal else FullINIWPrior

        # test common to natural
        my_emission_eta = conj_distr.common_to_natural(list(emisssion_post))
        my_emission_theta = conj_distr.natural_to_common(my_emission_eta)
        bnpy_emission_eta = get_bnpy_common_to_natural(bnpy_hmodel)
        params_name = ['tau', 'c', 'B', 'n']
        for i in range(len(params_name)):
            name = params_name[i]
            my_eta_val = my_emission_eta[i]
            bnpy_eta_val = bnpy_emission_eta[i]
            my_theta_val = my_emission_theta[i]
            origina_val = emisssion_post[i]
            if name == 'B' and is_diagonal:
                bnpy_eta_val = np.diagonal(bnpy_eta_val, axis1=-2, axis2=-1)

            assert th.all(th.isclose(my_eta_val, th.tensor(bnpy_eta_val).float())),\
                f"{m_type}: common_to_natural on {name} is not correct"

            assert th.all(th.isclose(my_theta_val, th.tensor(origina_val).float())), \
                f"{m_type}: natural_to_common on {name} is not correct"


def test_e_log_prec():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = create_data_and_bnpy_model(
            is_diagonal=is_diagonal)
        conj_distr = DiagonalNIWPrior if is_diagonal else FullINIWPrior
        eta = conj_distr.common_to_natural(list(emisssion_post))
        my_val = 2*conj_distr._exp_distr_class.expected_T_x(eta, 3)
        bnpy_val = get_bnpy_E_log_prec(bnpy_hmodel)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_perc is not correct"


def test_e_log_x():
    for m_type in ['diagonal', 'full']:
        is_diagonal = 'diagonal' == m_type
        bnpy_hmodel, x, stick_post, emisssion_post, stick_prior, emission_prior = create_data_and_bnpy_model(
            is_diagonal=is_diagonal)
        conj_distr = DiagonalNIWPrior if is_diagonal else FullINIWPrior
        eta = conj_distr.common_to_natural(list(emisssion_post))
        my_val = conj_distr.expected_data_loglikelihood(x, eta)
        bnpy_val = get_bnpy_E_log_x(bnpy_hmodel, x)
        assert th.all(th.isclose(my_val, th.tensor(bnpy_val).float())), f"{m_type}: E_log_x is not correct"