"""
Deterministic quanto correction Γ_t(U, T) — Equation (4.9) of the paper.

Custom implementation — Context7 found no library equivalent (paper Eq. 4.9).

The Vasicek loading functions are:
  sigma_S(t) = -sigma * n(t, S)          (domestic)
  sigma_hat_S(t) = -sigma_hat * n_hat(t, S)  (foreign)

where n(t, S) = (1 - exp(-b(S-t))) / b.

For t <= U:
  Γ_t(U, T) = exp(
        ∫_t^U (rho_12 * sigma_T(u) - rho_23 * sigma_bar) * sigma_hat_U(u) du
      - ∫_t^T (rho_12 * sigma_T(u) - rho_23 * sigma_bar) * sigma_hat_T(u) du
  )

For t >= U:
  Γ_t(U, T) = exp(
      - ∫_t^T (rho_12 * sigma_T(u) - rho_23 * sigma_bar) * sigma_hat_T(u) du
  )

We implement closed-form integrals for speed and reproducibility.
"""

from __future__ import annotations

import numpy as np

from src.params import ModelParams
from src.vasicek import n_func


# ---------------------------------------------------------------------------
# Vasicek loading functions  σ_S(t) and σ̂_S(t)
# ---------------------------------------------------------------------------

def sigma_S(t: float, S: float, p: ModelParams) -> float:
    """
    Domestic Vasicek loading function: σ_S(t) = -σ · n(t, S).

    Parameters
    ----------
    t : float  Current time.
    S : float  Target date (S >= t).
    p : ModelParams

    Returns
    -------
    float  σ_S(t) ≤ 0.
    """
    return -p.sigma * n_func(S - t, p.b)


def sigma_hat_S(t: float, S: float, p: ModelParams) -> float:
    """
    Foreign Vasicek loading function: σ̂_S(t) = -σ̂ · n̂(t, S).

    Parameters
    ----------
    t : float  Current time.
    S : float  Target date (S >= t).
    p : ModelParams

    Returns
    -------
    float  σ̂_S(t) ≤ 0.
    """
    return -p.sigma_hat * n_func(S - t, p.b_hat)


# ---------------------------------------------------------------------------
# Primitive integrals (closed-form, derived from Vasicek exponential structure)
# ---------------------------------------------------------------------------

def _int_n_hat(a_lo: float, a_hi: float, V: float, b_hat: float) -> float:
    """
    ∫_{a_lo}^{a_hi} n_hat(u, V) du = ∫ (1 - exp(-b_hat*(V-u))) / b_hat du

    Parameters
    ----------
    a_lo, a_hi : float  Integration limits.
    V          : float  Upper endpoint for n_hat.
    b_hat      : float  Foreign mean-reversion speed.
    """
    tau = a_hi - a_lo
    # ∫ (1 - exp(-b_hat*(V-u))) / b_hat du  over [a_lo, a_hi]
    # = (1/b_hat) * [tau - ∫ exp(-b_hat*(V-u)) du]
    # ∫ exp(-b_hat*(V-u)) du = [exp(-b_hat*(V-u))/b_hat]_{a_lo}^{a_hi}
    #                        = (exp(-b_hat*(V-a_hi)) - exp(-b_hat*(V-a_lo))) / b_hat
    return (1.0 / b_hat) * (
        tau - (np.exp(-b_hat * (V - a_hi)) - np.exp(-b_hat * (V - a_lo))) / b_hat
    )


def _int_n_n_hat(a_lo: float, a_hi: float, S: float, V: float,
                 b: float, b_hat: float) -> float:
    """
    ∫_{a_lo}^{a_hi} n(u, S) * n_hat(u, V) du

    = (1/(b*b_hat)) * ∫ (1-exp(-b*(S-u))) * (1-exp(-b_hat*(V-u))) du

    Expand and integrate term by term.
    """
    lo = a_lo
    hi = a_hi
    tau = hi - lo

    # ∫ 1 * 1 du = tau
    t1 = tau

    # ∫ exp(-b*(S-u)) du = [exp(-b*(S-u))/b]_{lo}^{hi}
    #                    = (exp(-b*(S-hi)) - exp(-b*(S-lo))) / b
    t2 = (np.exp(-b * (S - hi)) - np.exp(-b * (S - lo))) / b

    # ∫ exp(-b_hat*(V-u)) du  (analogous)
    t3 = (np.exp(-b_hat * (V - hi)) - np.exp(-b_hat * (V - lo))) / b_hat

    # ∫ exp(-b*(S-u)) * exp(-b_hat*(V-u)) du
    #  = exp(-b*S - b_hat*V) * ∫ exp((b+b_hat)*u) du
    #  = exp(-b*S - b_hat*V) * (exp((b+b_hat)*hi) - exp((b+b_hat)*lo)) / (b+b_hat)
    #  = (exp(-b*(S-hi) - b_hat*(V-hi)) - exp(-b*(S-lo) - b_hat*(V-lo))) / (b+b_hat)
    kk = b + b_hat
    t4 = (
        np.exp(-b * (S - hi) - b_hat * (V - hi))
        - np.exp(-b * (S - lo) - b_hat * (V - lo))
    ) / kk

    # Expansion: (1-e^{-b(S-u)})(1-e^{-b_hat(V-u)}) = 1 - e^{-b} - e^{-b_hat} + e^{-b-b_hat}
    return (t1 - t2 - t3 + t4) / (b * b_hat)


# ---------------------------------------------------------------------------
# Gamma correction — Equation (4.9)
# ---------------------------------------------------------------------------

def gamma_correction(t: float, U: float, T: float, p: ModelParams) -> float:
    """
    Compute Γ_t(U, T) from Equation (4.9).

    Parameters
    ----------
    t : float  Current time.
    U : float  Accrual start date (U >= t for the t<=U branch, otherwise t>=U).
    T : float  Accrual end / payment date (T > U).
    p : ModelParams

    Returns
    -------
    float  Γ_t(U, T) > 0.
    """
    rho12  = p.rho_12
    rho23  = p.rho_23
    sigma  = p.sigma
    s_hat  = p.sigma_hat
    sb     = p.sigma_bar
    b      = p.b
    b_hat  = p.b_hat

    # --- Second integral contribution: − ∫_t^T (ρ₁₂ σ_T − ρ₂₃ σ̄) σ̂_T du ---
    #
    # Expanding with σ_T(u)=−σ n(u,T) and σ̂_T(u)=−σ̂ n̂(u,T):
    #   (ρ₁₂·σ_T − ρ₂₃·σ̄)·σ̂_T
    #   = (ρ₁₂·(−σn) − ρ₂₃·σ̄)·(−σ̂n̂)
    #   = +ρ₁₂·σ·σ̂·n(u,T)·n̂(u,T) + ρ₂₃·σ̄·σ̂·n̂(u,T)
    #
    # The log-Gamma contribution is MINUS this integral:
    #   −ρ₁₂·σ·σ̂·I_nn_TT − ρ₂₃·σ̄·σ̂·I_n_hat_T

    I_nn_TT = _int_n_n_hat(t, T, T, T, b, b_hat)
    I_n_hat_T = _int_n_hat(t, T, T, b_hat)

    log_gamma_2 = -rho12 * sigma * s_hat * I_nn_TT - rho23 * sb * s_hat * I_n_hat_T

    if t <= U:
        # --- First integral contribution: + ∫_t^U (ρ₁₂ σ_T − ρ₂₃ σ̄) σ̂_U du ---
        #
        # (ρ₁₂·σ_T − ρ₂₃·σ̄)·σ̂_U
        # = (ρ₁₂·(−σ n(u,T)) − ρ₂₃·σ̄)·(−σ̂ n̂(u,U))
        # = +ρ₁₂·σ·σ̂·n(u,T)·n̂(u,U) + ρ₂₃·σ̄·σ̂·n̂(u,U)
        #
        # log-Gamma contribution is this integral directly:
        #   +ρ₁₂·σ·σ̂·I_nn_TU + ρ₂₃·σ̄·σ̂·I_n_hat_U

        I_nn_TU = _int_n_n_hat(t, U, T, U, b, b_hat)
        I_n_hat_U = _int_n_hat(t, U, U, b_hat)

        log_gamma_1 = rho12 * sigma * s_hat * I_nn_TU + rho23 * sb * s_hat * I_n_hat_U

        return np.exp(log_gamma_1 + log_gamma_2)
    else:
        return np.exp(log_gamma_2)


# ---------------------------------------------------------------------------
# Sensitivity utilities for exp_3
# ---------------------------------------------------------------------------

def gamma_correction_vary_rho23(t: float, U: float, T: float,
                                 p: ModelParams, rho23: float,
                                 sigma_bar: float) -> float:
    """
    Evaluate Γ_t(U, T) with overridden rho_23 and sigma_bar.

    The domestic-measure foreign drift c_hat is updated to keep the
    foreign-measure drift a_hat = c_hat + sigma_hat * sigma_bar * rho23 fixed.

    Per the paper: a_hat is held fixed at 0.05799 when varying rho_23.
    So: c_hat = a_hat - sigma_hat * sigma_bar * rho23.

    Parameters
    ----------
    t, U, T     : float  Time indices.
    p           : ModelParams  Baseline parameters.
    rho23       : float  New rho_23 value.
    sigma_bar   : float  New FX volatility.

    Returns
    -------
    float  Γ_t(U, T) with updated parameters.
    """
    from copy import copy
    p2 = copy(p)
    p2.rho_23 = rho23
    p2.sigma_bar = sigma_bar
    # a_hat (foreign-measure drift) = c_hat + sigma_hat * sigma_bar * rho23 = const 0.05799
    a_hat_fixed = 0.05799
    p2.c_hat = a_hat_fixed - p2.sigma_hat * sigma_bar * rho23
    return gamma_correction(t, U, T, p2)
