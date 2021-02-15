import numpy as np
from sklearn_m_ext import custom_metrics

y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_hat = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])


def test_csi_binary():
    exp = 0.5
    fun = custom_metrics.csi_score(y, y_hat)
    assert fun == exp


def test_bs_binary():
    exp = 1.1
    fun = custom_metrics.bs_score(y, y_hat)
    fun_round = round(fun, 1)
    assert fun_round == exp


def test_pss_binary():
    exp = 0.3
    fun = custom_metrics.pss_score(y, y_hat)
    fun_round = round(fun, 1)
    assert fun_round == exp


def test_or_binary():
    exp = 3.5
    fun = custom_metrics.or_score(y, y_hat)
    fun_round = round(fun, 1)
    assert fun_round == exp


def test_orss_binary():
    exp = 0.55556
    fun = custom_metrics.orss_score(y, y_hat)
    fun_round = round(fun, 5)
    assert fun_round == exp

