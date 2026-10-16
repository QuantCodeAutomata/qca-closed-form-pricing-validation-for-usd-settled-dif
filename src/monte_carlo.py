"""
Exact-Gaussian Monte Carlo pricing for USD-settled differential swaps.

Custom implementation — Context7 found no library equivalent for paper-specific
collateral-adjusted pricing with exact Vasicek joint distribution simulation.

Uses:
- Cholesky decomposition of the joint Gaussian covariance matrix.
- No Euler discretisation; exact conditional law at each coupon date.
- Common random numbers across compared strategies.

Monte Carlo price = A_{0,T}^{beta,i} * E[exp(-∫_0^T r^d du) * payoff]
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from src.params import ModelParams, ContractParams
from src.vasicek import (
    step_means, step_means_vec, step_cov_matrix,
    vasicek_mean, integrated_rate_mean,
)
from src.schedule import make_schedule, outstanding_coupons


# ---------------------------------------------------------------------------
# Joint distribution of (r^d, r^f, I^d, I^f) over multiple periods
# ---------------------------------------------------------------------------

def simulate_paths(
    dates: np.ndarray,
    p: ModelParams,
    n_paths: int = 250_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate exact joint Gaussian paths at each coupon date.

    For each period [T_{j-1}, T_j], given the state at T_{j-1}, draw
    (r^d_{T_j}, r^f_{T_j}, I^d_{T_{j-1},T_j}, I^f_{T_{j-1},T_j}) from
    the exact conditional Gaussian distribution using Cholesky decomposition.

    Parameters
    ----------
    dates   : np.ndarray  Schedule T_0, ..., T_n (length n+1).
    p       : ModelParams
    n_paths : int
    seed    : int

    Returns
    -------
    r_d   : np.ndarray  Shape (n_paths, n+1).  Domestic rates at each date.
    r_f   : np.ndarray  Shape (n_paths, n+1).  Foreign rates at each date.
    I_d   : np.ndarray  Shape (n_paths, n).    ∫ r^d du over each period.
    I_f   : np.ndarray  Shape (n_paths, n).    ∫ r^f du over each period.
    """
    n_dates = len(dates)
    n_periods = n_dates - 1

    rng = np.random.default_rng(seed)

    r_d = np.empty((n_paths, n_dates))
    r_f = np.empty((n_paths, n_dates))
    I_d = np.empty((n_paths, n_periods))
    I_f = np.empty((n_paths, n_periods))

    # Initialise at t=0
    r_d[:, 0] = p.r_d_0
    r_f[:, 0] = p.r_f_0

    for j, (t_prev, t_next) in enumerate(zip(dates[:-1], dates[1:])):
        tau = t_next - t_prev
        L = step_chol_matrix(tau, p)                    # (4,4) lower Chol

        # Standard normals
        Z = rng.standard_normal((n_paths, 4))           # shape (N, 4)
        perturb = Z @ L.T                               # shape (N, 4)

        # Compute conditional means (vectorised over paths)
        m = step_means_vec(r_d[:, j], r_f[:, j], tau, p)  # (N, 4)

        r_d[:, j + 1] = m[:, 0] + perturb[:, 0]
        r_f[:, j + 1] = m[:, 1] + perturb[:, 1]
        I_d[:, j]     = m[:, 2] + perturb[:, 2]
        I_f[:, j]     = m[:, 3] + perturb[:, 3]

    return r_d, r_f, I_d, I_f


def step_chol_matrix(tau: float, p: ModelParams) -> np.ndarray:
    """Compute and cache the Cholesky factor for a step of length tau."""
    C = step_cov_matrix(tau, p)
    C = 0.5 * (C + C.T)
    # Ensure positive-definiteness
    min_eig = np.min(np.linalg.eigvalsh(C))
    if min_eig < 0:
        raise ValueError(f"Covariance matrix not PSD: min eigenvalue {min_eig:.2e}")
    return cholesky(C, lower=True)


# ---------------------------------------------------------------------------
# Monte Carlo pricing for single-period coupon (Proposition 4.1 validation)
# ---------------------------------------------------------------------------

def mc_single_period(
    t0: float, U: float, T: float,
    c: ContractParams, p: ModelParams,
    n_paths: int = 250_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo price for a single-period coupon DS_0^{beta,gamma,kappa}(U, T; i).

    For the case t0 = 0 = T_0 (evaluation at inception).

    The discounted payoff is:
      D_T * A_{0,T}^{beta,i} * [exp(I^d_{U,T}) - gamma*exp(I^f_{U,T}) - (1-gamma+delta*kappa)]

    where D_T = exp(-∫_0^T r^d_u du) = exp(-I^d_{0,T}).

    Parameters
    ----------
    t0     : float  Start time (should be 0.0).
    U      : float  Accrual start (>= t0).
    T      : float  Payment date.
    c      : ContractParams
    p      : ModelParams
    n_paths: int
    seed   : int

    Returns
    -------
    (price, stderr) : tuple[float, float]
    """
    assert t0 == 0.0, "MC single-period implemented for t0=0 only"
    assert U >= t0

    delta = T - U
    alpha = p.alpha_beta(c.beta, c.collateral)

    # Build schedule: T_0=0, then U (if U>0), then T
    if U > t0 + 1e-12:
        dates = np.array([t0, U, T])
    else:
        dates = np.array([t0, T])

    r_d, r_f, I_d, I_f = simulate_paths(dates, p, n_paths=n_paths, seed=seed)

    # I^d_{0,T} = sum of I_d periods up to T
    I_d_total = np.sum(I_d, axis=1)          # ∫_0^T r^d du

    # I^d_{U,T} and I^f_{U,T}
    if U > t0 + 1e-12:
        # dates = [0, U, T]:  I_d[:,0] = I^d_{0,U},  I_d[:,1] = I^d_{U,T}
        I_d_UT = I_d[:, 1]
        I_f_UT = I_f[:, 1]
    else:
        # U = 0 = t0: dates = [0, T], I_d[:,0] = I^d_{0,T}
        I_d_UT = I_d[:, 0]
        I_f_UT = I_f[:, 0]

    # Discount factor
    D_T = np.exp(-I_d_total)
    A = np.exp(-alpha * T)

    # Payoff at T
    payoff = np.exp(I_d_UT) - c.gamma * np.exp(I_f_UT) - (1.0 - c.gamma + delta * c.kappa)

    # Discounted payoff
    discounted = D_T * A * payoff

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(stderr)


# ---------------------------------------------------------------------------
# Monte Carlo pricing for multi-period swap (Proposition 4.2 validation)
# ---------------------------------------------------------------------------

def mc_multi_period(
    dates: np.ndarray,
    deltas: np.ndarray,
    c: ContractParams, p: ModelParams,
    n_paths: int = 250_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo price for a multi-period USD-settled differential swap.

    The time-0 value equals the sum over all coupons:
      Sum_j A_{0,T_j}^{beta,i} * E[ exp(-I^d_{0,T_j}) * coupon_payoff_j ]

    where coupon_payoff_j = exp(I^d_{T_{j-1},T_j}) - gamma*exp(I^f_{T_{j-1},T_j})
                           - (1 - gamma + delta_j*kappa)

    Parameters
    ----------
    dates  : np.ndarray  T_0..T_n.
    deltas : np.ndarray  Accrual fractions.
    c      : ContractParams
    p      : ModelParams
    n_paths: int
    seed   : int

    Returns
    -------
    (price, stderr) : tuple[float, float]
    """
    alpha = p.alpha_beta(c.beta, c.collateral)

    r_d, r_f, I_d, I_f = simulate_paths(dates, p, n_paths=n_paths, seed=seed)

    n_coupons = len(deltas)
    # Cumulative ∫_0^{T_j} r^d du for each coupon payment date
    I_d_cum = np.cumsum(I_d, axis=1)   # shape (n_paths, n_coupons)

    total_payoff = np.zeros(n_paths)
    for j in range(n_coupons):
        T_j = dates[j + 1]
        delta_j = deltas[j]
        A_j = np.exp(-alpha * T_j)

        D_Tj = np.exp(-I_d_cum[:, j])                   # exp(-∫_0^{T_j} r^d du)
        coupon = (
            np.exp(I_d[:, j])
            - c.gamma * np.exp(I_f[:, j])
            - (1.0 - c.gamma + delta_j * c.kappa)
        )
        total_payoff += A_j * D_Tj * coupon

    price = total_payoff.mean()
    stderr = total_payoff.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(stderr)


# ---------------------------------------------------------------------------
# Validation table helper
# ---------------------------------------------------------------------------

def mc_validation_table(
    analytical: float,
    mc_price: float,
    mc_se: float,
    label: str = "",
) -> dict:
    """
    Compute z-score and error for a single validation case.

    Returns dict with keys: label, analytical, mc, error, se, z_score.
    """
    error = mc_price - analytical
    z = error / mc_se if mc_se > 0 else float("nan")
    return {
        "label": label,
        "analytical": analytical,
        "mc": mc_price,
        "error": error,
        "se": mc_se,
        "z_score": z,
    }
