"""
Analytical pricing of USD-settled differential swaps.

Custom implementation — Context7 found no library equivalent (paper-specific
collateral convention and quanto correction).

Implements:
  - Proposition 4.1: single-period coupon value DS_t^{beta,gamma,kappa}(U, T; i)
  - Proposition 4.2: multi-period swap value (sum over outstanding coupons)
  - Corollary 4.1:   par margin kappa_0^*(T_0, n; beta, gamma, i)
"""

from __future__ import annotations

import numpy as np

from src.params import ModelParams, ContractParams
from src.vasicek import bond_d, futures_f, n_func
from src.quanto import gamma_correction
from src.schedule import make_schedule, outstanding_coupons


# ---------------------------------------------------------------------------
# Collateral discount multiplier
# ---------------------------------------------------------------------------

def collateral_multiplier(t: float, T: float,
                           beta: float, collateral: str,
                           p: ModelParams) -> float:
    """
    A_{t,T}^{beta,i} = exp(-alpha^{beta,i} * (T - t))   (constant spread case).

    Parameters
    ----------
    t, T       : float  Current and payment time.
    beta       : float  Collateralisation fraction.
    collateral : str    'USD' or 'EUR'.
    p          : ModelParams

    Returns
    -------
    float  A_{t,T}^{beta,i}.
    """
    alpha = p.alpha_beta(beta, collateral)
    return np.exp(-alpha * (T - t))


# ---------------------------------------------------------------------------
# Proposition 4.1 building blocks — I^(1), I^(2), I^(3)
# ---------------------------------------------------------------------------

def coupon_I1(t: float, U: float, T: float,
              r_d: float, beta: float, collateral: str,
              p: ModelParams) -> float:
    """
    I_t^{(1)}(U, T) from Proposition 4.1.

    For t <= U:  A_{t,T}^{beta,i} * B^d(t, U)
    For U < t <= T:  A_{t,T}^{beta,i} * B^d(U,U)^{-1} * B^d(t,...}
                     -> analytically A_{t,T}^{beta,i} * exp(+integral_U^t r_u du)

    Since we evaluate at t=0 only (U >= 0), this reduces to:
      I_0^{(1)} = A_{0,T}^{beta,i} * B^d(0, U)

    At t=0, U=0:  B^d(0, 0) = 1.
    """
    assert t <= T, "t must be <= T"
    A = collateral_multiplier(t, T, beta, collateral, p)
    if t <= U:
        bd = bond_d(r_d, U - t, p) if U > t else 1.0
        return A * bd
    else:
        # ex-dividend branch (t in (U, T]): A_{t,T} * B^d(t,T)/B^d(U,T)... simplified
        # This branch applies for t > U; we return the ex-coupon value.
        # The Proposition says: A_{t,T}^{beta,i} * (B_U^d)^{-1} * B_t^d
        # Under Vasicek B_t^d = B^d(t, T) and (B_U^d)^{-1} = (B^d(U,T))^{-1}... wait
        # The paper notation B_t^d = B^d(0,t) is the money-market account.
        # After the accrual start U, the term I^(1) uses the money-market ratio.
        # I^(1) = A * (B^d(t)/B^d(U))  where B^d(t) = exp(∫_0^t r du) is the bank account.
        # At t=0 init this branch doesn't apply; for t > U we raise an error.
        raise ValueError("ex-dividend branch (t > U) not used in exp_1 at t=0")


def coupon_I2(t: float, U: float, T: float,
              r_d: float, r_f: float,
              beta: float, collateral: str,
              p: ModelParams, gamma_val: float | None = None) -> float:
    """
    I_t^{(2)}(U, T) = A_{t,T}^{beta,i} * B^d(t, T) * (1 + delta * F_t^f(U, T)) * Gamma_t(U, T).

    Parameters
    ----------
    gamma_val : float | None  If provided, override Gamma (e.g., set to 1 for no-quanto case).
    """
    delta = T - U
    A = collateral_multiplier(t, T, beta, collateral, p)
    bd = bond_d(r_d, T - t, p)
    Ff = futures_f(r_f, t, U, T, p)
    if gamma_val is None:
        G = gamma_correction(t, U, T, p)
    else:
        G = gamma_val
    return A * bd * (1.0 + delta * Ff) * G


def coupon_I3(t: float, T: float,
              r_d: float, beta: float, collateral: str,
              p: ModelParams) -> float:
    """
    I_t^{(3)}(T) = A_{t,T}^{beta,i} * B^d(t, T).
    """
    A = collateral_multiplier(t, T, beta, collateral, p)
    bd = bond_d(r_d, T - t, p)
    return A * bd


# ---------------------------------------------------------------------------
# Proposition 4.1 — single-period coupon
# ---------------------------------------------------------------------------

def single_period_value(
    t: float, U: float, T: float,
    r_d: float, r_f: float,
    c: ContractParams, p: ModelParams,
    gamma_override: float | None = None,
) -> float:
    """
    DS_t^{beta,gamma,kappa}(U, T; i)  — Proposition 4.1.

    = I_t^{(1)} - gamma * I_t^{(2)} - (1 - gamma + delta * kappa) * I_t^{(3)}

    Parameters
    ----------
    t           : float  Current time (t <= T).
    U           : float  Accrual start.
    T           : float  Payment / accrual end.
    r_d, r_f   : float  Current short rates.
    c           : ContractParams
    p           : ModelParams
    gamma_override : float | None  Override for the Gamma function (set to 1 for no-quanto).

    Returns
    -------
    float  Per-unit-notional swap value.
    """
    delta = T - U
    I1 = coupon_I1(t, U, T, r_d, c.beta, c.collateral, p)
    I2 = coupon_I2(t, U, T, r_d, r_f, c.beta, c.collateral, p, gamma_val=gamma_override)
    I3 = coupon_I3(t, T, r_d, c.beta, c.collateral, p)
    return I1 - c.gamma * I2 - (1.0 - c.gamma + delta * c.kappa) * I3


# ---------------------------------------------------------------------------
# Proposition 4.2 — multi-period swap (sum over outstanding coupons)
# ---------------------------------------------------------------------------

def multi_period_value(
    t: float,
    dates: np.ndarray,
    deltas: np.ndarray,
    r_d: float, r_f: float,
    c: ContractParams, p: ModelParams,
    gamma_override: float | None = None,
) -> float:
    """
    DS_t^{beta,gamma,kappa}(T_0, n; i)  — Proposition 4.2.

    Sum over outstanding coupons j: I_j^{(1)} - gamma*I_j^{(2)} - (1-gamma+delta_j*kappa)*I_j^{(3)}.

    Parameters
    ----------
    t      : float       Current time.
    dates  : np.ndarray  Coupon dates T_0..T_n (length n+1).
    deltas : np.ndarray  Accrual fractions δ_j (length n).
    r_d    : float       Domestic short rate at t.
    r_f    : float       Foreign short rate at t.
    c      : ContractParams
    p      : ModelParams
    gamma_override : float | None  Override for Gamma in all coupons.

    Returns
    -------
    float  Per-unit-notional swap value (sum over outstanding coupons).
    """
    outstanding = outstanding_coupons(t, dates)
    total = 0.0
    for j in outstanding:
        U_j = dates[j]      # accrual start = dates[j]  (T_{j-1} in 0-indexed terms)
        T_j = dates[j + 1]  # payment = dates[j+1]      (T_j in 0-indexed terms)
        delta_j = deltas[j]

        # Create a local coupon contract with the same settings
        c_local = ContractParams(
            gamma=c.gamma, kappa=c.kappa, beta=c.beta,
            collateral=c.collateral, notional=c.notional,
        )
        total += single_period_value(t, U_j, T_j, r_d, r_f, c_local, p, gamma_override)
    return total


# ---------------------------------------------------------------------------
# Corollary 4.1 — par margin
# ---------------------------------------------------------------------------

def par_margin(
    t: float,
    dates: np.ndarray,
    deltas: np.ndarray,
    r_d: float, r_f: float,
    gamma: int, beta: float, collateral: str,
    p: ModelParams,
    gamma_override: float | None = None,
) -> float:
    """
    kappa_0^*(T_0, n; beta, gamma, i)  — Corollary 4.1.

    kappa^* = [sum_j (I_j^{(1)} - gamma*I_j^{(2)} - (1-gamma)*I_j^{(3)})]
              / [sum_j delta_j * I_j^{(3)}]

    Parameters
    ----------
    t          : float  Current time.
    dates      : np.ndarray
    deltas     : np.ndarray
    r_d, r_f   : float  Short rates at t.
    gamma      : int    0 or 1.
    beta       : float  Collateralisation fraction.
    collateral : str    'USD' or 'EUR'.
    p          : ModelParams
    gamma_override : float | None  Override Gamma to 1 for no-quanto counterfactual.

    Returns
    -------
    float  Par margin in decimal (e.g., 0.04 = 4%).
    """
    outstanding = outstanding_coupons(t, dates)
    numerator = 0.0
    denominator = 0.0

    for j in outstanding:
        U_j = dates[j]
        T_j = dates[j + 1]
        delta_j = deltas[j]

        I1 = coupon_I1(t, U_j, T_j, r_d, beta, collateral, p)
        I2 = coupon_I2(t, U_j, T_j, r_d, r_f, beta, collateral, p, gamma_val=gamma_override)
        I3 = coupon_I3(t, T_j, r_d, beta, collateral, p)

        numerator += I1 - gamma * I2 - (1.0 - gamma) * I3
        denominator += delta_j * I3

    return numerator / denominator
