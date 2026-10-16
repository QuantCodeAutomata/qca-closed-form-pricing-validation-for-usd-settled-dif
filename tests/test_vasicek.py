"""
Unit tests for Vasicek utilities (src/vasicek.py).
"""

import numpy as np
import pytest
from src.params import ModelParams
from src.vasicek import (
    n_func, vasicek_bond_log, bond_d, bond_f,
    vasicek_mean, vasicek_var_r,
    integrated_rate_mean, integrated_rate_var, cov_r_integrated,
    cov_r1_r2, cov_r1_I2, cov_I1_I2,
    step_cov_matrix, step_means_vec,
    futures_d, futures_f,
)

P = ModelParams()


# ---------------------------------------------------------------------------
# n_func
# ---------------------------------------------------------------------------

def test_n_func_zero_tau():
    assert n_func(0.0, 3.0) == pytest.approx(0.0, abs=1e-12)


def test_n_func_limit_k_zero():
    """As k -> 0, n(tau; k) -> tau."""
    val = n_func(2.0, 1e-10)
    assert val == pytest.approx(2.0, rel=1e-5)


def test_n_func_positive():
    assert n_func(1.0, 3.0) > 0


def test_n_func_monotone_in_tau():
    assert n_func(2.0, 3.0) > n_func(1.0, 3.0)


# ---------------------------------------------------------------------------
# Bond prices
# ---------------------------------------------------------------------------

def test_bond_d_at_zero_tau():
    """B^d(t,t) = 1."""
    assert bond_d(0.03, 0.0, P) == pytest.approx(1.0, abs=1e-12)


def test_bond_d_positive():
    assert bond_d(P.r_d_0, 1.0, P) > 0


def test_bond_d_less_than_one():
    """Bond price < 1 for positive rates and tau > 0."""
    assert bond_d(P.r_d_0, 1.0, P) < 1.0


def test_bond_d_higher_rate_lower_price():
    """Higher short rate => lower bond price."""
    b1 = bond_d(0.01, 2.0, P)
    b2 = bond_d(0.10, 2.0, P)
    assert b1 > b2


def test_bond_f_at_zero_tau():
    assert bond_f(0.02, 0.0, P) == pytest.approx(1.0, abs=1e-12)


def test_bond_f_positive():
    assert bond_f(P.r_f_0, 0.5, P) > 0


# ---------------------------------------------------------------------------
# Conditional moments
# ---------------------------------------------------------------------------

def test_vasicek_mean_at_t0():
    """E[r_t | r_0] at tau=0 equals r_0."""
    assert vasicek_mean(0.03, 0.0, 0.04, 3.0) == pytest.approx(0.03, abs=1e-12)


def test_vasicek_mean_reverts():
    """Mean should revert toward theta for large tau."""
    theta = P.a / P.b
    m = vasicek_mean(P.r_d_0, 100.0, theta, P.b)
    assert abs(m - theta) < 1e-4


def test_vasicek_var_zero_tau():
    assert vasicek_var_r(0.0, P.b, P.sigma) == pytest.approx(0.0, abs=1e-12)


def test_vasicek_var_positive():
    assert vasicek_var_r(1.0, P.b, P.sigma) > 0


def test_integrated_rate_mean_zero_tau():
    """E[∫_0^0 r du] = 0."""
    assert integrated_rate_mean(0.05, 0.0, 0.04, 3.0) == pytest.approx(0.0, abs=1e-12)


def test_integrated_rate_var_zero_tau():
    assert integrated_rate_var(0.0, P.b, P.sigma) == pytest.approx(0.0, abs=1e-12)


def test_cov_r_integrated_zero_tau():
    assert cov_r_integrated(0.0, P.b, P.sigma) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def test_step_cov_matrix_psd():
    """4×4 covariance matrix is positive semi-definite."""
    C = step_cov_matrix(0.5, P)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals >= -1e-10), f"Min eigenvalue {eigvals.min():.2e}"


def test_step_cov_matrix_symmetric():
    C = step_cov_matrix(0.5, P)
    np.testing.assert_allclose(C, C.T, atol=1e-15)


def test_step_cov_matrix_zero_tau():
    """Covariance at tau=0 should be zero matrix."""
    C = step_cov_matrix(0.0, P)
    np.testing.assert_allclose(C, np.zeros((4, 4)), atol=1e-12)


# ---------------------------------------------------------------------------
# Futures expectations
# ---------------------------------------------------------------------------

def test_futures_f_positive_rate():
    """F^f(U,T) for positive foreign rate should be positive."""
    Ff = futures_f(P.r_f_0, 0.0, 0.0, 0.5, P)
    assert Ff > 0


def test_futures_d_positive_rate():
    """F^d(U,T) for positive domestic rate should be positive."""
    Fd = futures_d(P.r_d_0, 0.0, 0.0, 0.5, P)
    assert Fd > 0


def test_futures_f_approaches_spot():
    """For short maturity, futures ≈ current short rate."""
    Ff = futures_f(0.05, 0.0, 0.0, 0.001, P)
    assert abs(Ff - 0.05) < 0.01


def test_futures_d_approaches_spot():
    Fd = futures_d(0.03, 0.0, 0.0, 0.001, P)
    assert abs(Fd - 0.03) < 0.01


def test_futures_f_with_future_accrual():
    """F^f for U>0 (forward-starting)."""
    Ff = futures_f(P.r_f_0, 0.0, 1.0, 1.5, P)
    assert Ff > 0


# ---------------------------------------------------------------------------
# Step mean vector
# ---------------------------------------------------------------------------

def test_step_means_vec_shape():
    from src.monte_carlo import step_means_vec
    r_d = np.array([0.03, 0.04, 0.05])
    r_f = np.array([0.02, 0.02, 0.02])
    means = step_means_vec(r_d, r_f, 0.5, P)
    assert means.shape == (3, 4)


def test_step_means_vec_zero_tau():
    from src.monte_carlo import step_means_vec
    r_d = np.array([0.03])
    r_f = np.array([0.02])
    means = step_means_vec(r_d, r_f, 0.0, P)
    assert means[0, 0] == pytest.approx(0.03, abs=1e-10)
    assert means[0, 1] == pytest.approx(0.02, abs=1e-10)
    assert means[0, 2] == pytest.approx(0.0, abs=1e-10)
    assert means[0, 3] == pytest.approx(0.0, abs=1e-10)
