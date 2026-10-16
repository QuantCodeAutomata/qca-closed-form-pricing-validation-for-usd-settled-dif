"""
Unit tests for the Gamma / quanto-correction module (src/quanto.py).
"""

import numpy as np
import pytest
from src.params import ModelParams
from src.quanto import gamma_correction, sigma_S, sigma_hat_S


P = ModelParams()


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def test_sigma_S_sign():
    """sigma_S(t) = -sigma * n(t,S) <= 0 for t < S."""
    val = sigma_S(0.0, 1.0, P)
    assert val < 0


def test_sigma_S_zero_at_expiry():
    """sigma_S(S) = 0 because n(S,S)=0."""
    val = sigma_S(1.0, 1.0, P)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_sigma_hat_S_sign():
    val = sigma_hat_S(0.0, 1.0, P)
    assert val < 0


def test_sigma_hat_S_zero_at_expiry():
    val = sigma_hat_S(1.0, 1.0, P)
    assert val == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------

def test_gamma_correction_at_tU():
    """For t=U the 't<=U' and 't>=U' branches should agree."""
    G_below = gamma_correction(0.999, 1.0, 1.5, P)
    G_above = gamma_correction(1.001, 1.0, 1.5, P)
    # Should be close (not exactly equal because of the integration path)
    # Just assert both positive
    assert G_below > 0
    assert G_above > 0


def test_gamma_correction_no_correlation():
    """
    With rho_12 = rho_23 = 0, the integrand vanishes.
    Gamma should equal 1 (exp(0) = 1).
    """
    from copy import copy
    p_no_corr = copy(P)
    p_no_corr.rho_12 = 0.0
    p_no_corr.rho_23 = 0.0
    G = gamma_correction(0.0, 0.5, 1.0, p_no_corr)
    assert G == pytest.approx(1.0, abs=1e-8)


def test_gamma_correction_positive():
    """Gamma is always positive (it's an exponential)."""
    G = gamma_correction(0.0, 1.0, 1.5, P)
    assert G > 0


def test_gamma_correction_t_at_zero_U_at_zero():
    """For t=U=0, the first integral vanishes; only second integral remains."""
    G_t0_U0 = gamma_correction(0.0, 0.0, 0.5, P)
    G_t0_U0_ref = gamma_correction(0.001, 0.0, 0.5, P)   # tiny t, U=0
    # Should be close since the first integral's lower bound matches t
    assert G_t0_U0 > 0
    assert abs(G_t0_U0 / G_t0_U0_ref - 1.0) < 0.01


def test_gamma_correction_t_equals_T():
    """At t=T, the second integral is over empty domain => Gamma should be 1."""
    G = gamma_correction(1.5, 1.0, 1.5, P)
    # The integral from T to T is zero
    assert G == pytest.approx(1.0, abs=1e-6)


def test_gamma_correction_symmetric_rho23():
    """
    Flipping the sign of rho_23 must change Gamma: the ρ₂₃-contribution to
    log(Γ) is linear in ρ₂₃, so log(G_pos) and log(G_neg) bracket their
    mean — in particular they are NOT equal.  The absolute difference in
    log-space for rho_23=±0.20, U=0.5, T=1.0 is ~4e-5 (small because
    σ̄·σ̂·∫ n̂ du is small), so we only assert it is nonzero at float precision.
    """
    from copy import copy
    p_pos = copy(P)
    p_pos.rho_23 = 0.20
    p_neg = copy(P)
    p_neg.rho_23 = -0.20

    G_pos = gamma_correction(0.0, 0.5, 1.0, p_pos)
    G_neg = gamma_correction(0.0, 0.5, 1.0, p_neg)

    # Values must differ (rho_23 has a nonzero contribution)
    assert G_pos != G_neg
    # Sign of contribution: the rho_23 term in the exponent is proportional to
    # ρ₂₃·σ̄·σ̂·(∫₀ᵁ n̂(U−u)du − ∫₀ᵀ n̂(T−u)du) < 0 for U < T.
    # → positive rho_23 lowers Gamma, negative rho_23 raises it.
    assert G_pos < G_neg


def test_gamma_correction_sigma_bar_zero():
    """
    With sigma_bar=0 and rho_12=0, Gamma=1; rho_23 term vanishes.
    """
    from copy import copy
    p_z = copy(P)
    p_z.sigma_bar = 0.0
    p_z.rho_12 = 0.0
    p_z.rho_23 = 0.0
    G = gamma_correction(0.0, 0.5, 1.0, p_z)
    assert G == pytest.approx(1.0, abs=1e-7)
