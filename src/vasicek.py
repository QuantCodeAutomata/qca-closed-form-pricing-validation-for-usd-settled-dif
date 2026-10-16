"""
Vasicek affine term-structure utilities.

Custom implementation — Context7 found no library equivalent for paper-specific
collateral-adjusted pricing with quanto correction (custom per-paper formulas).

Provides:
- Zero-coupon bond prices B^d(t,T) and B^f notation helpers.
- Conditional mean / variance / covariance of the Vasicek state.
- Domestic-measure foreign-rate futures expectation F^f_t(U,T).
- Domestic futures expectation F^d_t(U,T).
- 4×4 covariance matrix for exact Gaussian one-step simulation
  of (r^d, r^f, I^d, I^f) over a single period.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from src.params import ModelParams


# ---------------------------------------------------------------------------
# Vasicek loading / discount functions
# ---------------------------------------------------------------------------

def n_func(tau: float, k: float) -> float:
    """
    n(tau; k) = (1 - exp(-k*tau)) / k

    The duration loading from the Vasicek bond-price formula.
    Equals tau as k → 0.

    Parameters
    ----------
    tau : float  Time to maturity (T - t) in years.
    k   : float  Mean-reversion speed (b or b_hat).
    """
    if k == 0.0:
        return tau
    return (1.0 - np.exp(-k * tau)) / k


def vasicek_bond_log(r0: float, tau: float,
                     theta: float, k: float, sigma: float) -> float:
    """
    log B(t, t+tau) = A(tau) - n(tau; k) * r0

    Standard Vasicek closed-form bond price (under the corresponding
    risk-neutral measure).

    Parameters
    ----------
    r0    : float  Short rate at time t.
    tau   : float  Time to maturity.
    theta : float  Long-run mean = a/b (or c_hat/b_hat).
    k     : float  Mean-reversion speed.
    sigma : float  Short-rate volatility.

    Returns
    -------
    float  log bond price.
    """
    n = n_func(tau, k)
    # A(tau) = (n - tau) * (theta - sigma^2/(2k^2)) - sigma^2 * n^2 / (4k)
    A = (n - tau) * (theta - sigma ** 2 / (2.0 * k ** 2)) - sigma ** 2 * n ** 2 / (4.0 * k)
    return A - n * r0


def bond_d(r_d: float, tau: float, p: ModelParams) -> float:
    """
    B^d(t, t+tau): domestic zero-coupon bond price.

    Parameters
    ----------
    r_d : float  Domestic short rate at t.
    tau : float  Time to maturity.
    p   : ModelParams
    """
    theta_d = p.a / p.b
    return np.exp(vasicek_bond_log(r_d, tau, theta_d, p.b, p.sigma))


def bond_f(r_f: float, tau: float, p: ModelParams) -> float:
    """
    B^f(t, t+tau): foreign zero-coupon bond price *under the domestic measure Q_d*.

    Under Q_d, r^f follows: dr^f = (c_hat - b_hat r^f) dt + sigma_hat dZ^2.
    Long-run mean = c_hat / b_hat.

    Parameters
    ----------
    r_f : float  Foreign short rate at t.
    tau : float  Time to maturity.
    p   : ModelParams
    """
    theta_f = p.c_hat / p.b_hat
    return np.exp(vasicek_bond_log(r_f, tau, theta_f, p.b_hat, p.sigma_hat))


# ---------------------------------------------------------------------------
# Conditional moments of a single Vasicek factor
# ---------------------------------------------------------------------------

def vasicek_mean(r0: float, tau: float, theta: float, k: float) -> float:
    """E[r_{t+tau} | r_t = r0]."""
    return r0 * np.exp(-k * tau) + theta * (1.0 - np.exp(-k * tau))


def vasicek_var_r(tau: float, k: float, sigma: float) -> float:
    """Var[r_{t+tau} | r_t]."""
    return sigma ** 2 * (1.0 - np.exp(-2.0 * k * tau)) / (2.0 * k)


def integrated_rate_mean(r0: float, tau: float, theta: float, k: float) -> float:
    """E[∫_t^{t+tau} r_u du | r_t = r0]."""
    n = n_func(tau, k)
    return (r0 - theta) * n + theta * tau


def integrated_rate_var(tau: float, k: float, sigma: float) -> float:
    """Var[∫_t^{t+tau} r_u du | r_t]."""
    n = n_func(tau, k)
    n2 = n_func(tau, 2.0 * k)
    # (sigma/k)^2 * [tau - 2*n + n2/2] -- careful with n2 for 2k
    # n(tau;2k) = (1-exp(-2k*tau))/(2k)
    return (sigma / k) ** 2 * (tau - 2.0 * n + (1.0 - np.exp(-2.0 * k * tau)) / (2.0 * k))


def cov_r_integrated(tau: float, k: float, sigma: float) -> float:
    """Cov[r_{t+tau}, ∫_t^{t+tau} r_u du | r_t]."""
    n = n_func(tau, k)
    n2 = (1.0 - np.exp(-2.0 * k * tau)) / (2.0 * k)
    return sigma ** 2 / k * (n - n2)


# ---------------------------------------------------------------------------
# Cross-covariance between two Vasicek factors (with correlation rho)
# ---------------------------------------------------------------------------

def cov_r1_r2(tau: float, k1: float, s1: float,
              k2: float, s2: float, rho: float) -> float:
    """Cov[r1_{t+tau}, r2_{t+tau}] with mean-rev speeds k1,k2, vols s1,s2."""
    return rho * s1 * s2 * (1.0 - np.exp(-(k1 + k2) * tau)) / (k1 + k2)


def cov_r1_I2(tau: float, k1: float, s1: float,
              k2: float, s2: float, rho: float) -> float:
    """Cov[r1_{t+tau}, ∫_t^{t+tau} r2_u du]."""
    # = rho*s1*s2/k2 * [ (1-exp(-k2*tau))/k2 - (1-exp(-(k1+k2)*tau))/(k1+k2) ]
    return rho * s1 * s2 / k2 * (
        (1.0 - np.exp(-k2 * tau)) / k2
        - (1.0 - np.exp(-(k1 + k2) * tau)) / (k1 + k2)
    )


def cov_I1_I2(tau: float, k1: float, s1: float,
              k2: float, s2: float, rho: float) -> float:
    """Cov[∫_t^{t+tau} r1_u du, ∫_t^{t+tau} r2_u du]."""
    # = rho*s1*s2/(k1*k2) * [tau - n(tau;k1) - n(tau;k2) + n(tau;k1+k2)]
    n1 = n_func(tau, k1)
    n2 = n_func(tau, k2)
    n12 = n_func(tau, k1 + k2)
    return rho * s1 * s2 / (k1 * k2) * (tau - n1 - n2 + n12)


# ---------------------------------------------------------------------------
# 4×4 joint covariance matrix for one-step exact Gaussian simulation
# State vector: X = (r^d, r^f, I^d, I^f) over period of length tau
# ---------------------------------------------------------------------------

def step_cov_matrix(tau: float, p: ModelParams) -> np.ndarray:
    """
    4×4 conditional covariance matrix of (r^d, r^f, I^d, I^f) over [0, tau]
    given (r^d_0, r^f_0).

    Indexing: 0=r^d, 1=r^f, 2=I^d, 3=I^f.

    Parameters
    ----------
    tau : float  Step length.
    p   : ModelParams

    Returns
    -------
    np.ndarray  Shape (4,4), symmetric PSD.
    """
    C = np.zeros((4, 4))

    # Diagonal
    C[0, 0] = vasicek_var_r(tau, p.b, p.sigma)
    C[1, 1] = vasicek_var_r(tau, p.b_hat, p.sigma_hat)
    C[2, 2] = integrated_rate_var(tau, p.b, p.sigma)
    C[3, 3] = integrated_rate_var(tau, p.b_hat, p.sigma_hat)

    # Off-diagonal (symmetric)
    C[0, 2] = C[2, 0] = cov_r_integrated(tau, p.b, p.sigma)
    C[1, 3] = C[3, 1] = cov_r_integrated(tau, p.b_hat, p.sigma_hat)

    # Cross-factor (domestic vs foreign, correlation rho_12)
    C[0, 1] = C[1, 0] = cov_r1_r2(tau, p.b, p.sigma, p.b_hat, p.sigma_hat, p.rho_12)
    C[0, 3] = C[3, 0] = cov_r1_I2(tau, p.b, p.sigma, p.b_hat, p.sigma_hat, p.rho_12)
    C[1, 2] = C[2, 1] = cov_r1_I2(tau, p.b_hat, p.sigma_hat, p.b, p.sigma, p.rho_12)
    C[2, 3] = C[3, 2] = cov_I1_I2(tau, p.b, p.sigma, p.b_hat, p.sigma_hat, p.rho_12)

    return C


def step_chol(tau: float, p: ModelParams) -> np.ndarray:
    """
    Lower-triangular Cholesky factor of step_cov_matrix.

    Using scipy.linalg.cholesky (lower=True).
    """
    C = step_cov_matrix(tau, p)
    # Symmetrize to handle any floating-point asymmetry
    C = 0.5 * (C + C.T)
    return cholesky(C, lower=True)


# ---------------------------------------------------------------------------
# Conditional means for one-step simulation
# ---------------------------------------------------------------------------

def step_means(r_d_prev: float, r_f_prev: float, tau: float, p: ModelParams) -> np.ndarray:
    """
    Mean vector of (r^d, r^f, I^d, I^f) over [0, tau] given current state.

    Parameters
    ----------
    r_d_prev : float  Domestic rate at start of period.
    r_f_prev : float  Foreign rate at start of period.
    tau      : float  Period length.
    p        : ModelParams

    Returns
    -------
    np.ndarray  Shape (4,).
    """
    theta_d = p.a / p.b
    theta_f = p.c_hat / p.b_hat

    m_rd = vasicek_mean(r_d_prev, tau, theta_d, p.b)
    m_rf = vasicek_mean(r_f_prev, tau, theta_f, p.b_hat)
    m_Id = integrated_rate_mean(r_d_prev, tau, theta_d, p.b)
    m_If = integrated_rate_mean(r_f_prev, tau, theta_f, p.b_hat)

    return np.array([m_rd, m_rf, m_Id, m_If])


def step_means_vec(r_d_prev: np.ndarray, r_f_prev: np.ndarray,
                   tau: float, p: ModelParams) -> np.ndarray:
    """
    Vectorised conditional means for a batch of initial states.

    Parameters
    ----------
    r_d_prev : np.ndarray  Shape (n_paths,). Domestic rates at start of period.
    r_f_prev : np.ndarray  Shape (n_paths,). Foreign rates at start of period.
    tau      : float       Period length.
    p        : ModelParams

    Returns
    -------
    np.ndarray  Shape (n_paths, 4):  columns (m_rd, m_rf, m_Id, m_If).
    """
    theta_d = p.a / p.b
    theta_f = p.c_hat / p.b_hat
    ed = np.exp(-p.b * tau)
    ef = np.exp(-p.b_hat * tau)
    n_d = (1.0 - ed) / p.b
    n_f = (1.0 - ef) / p.b_hat

    m_rd = r_d_prev * ed + theta_d * (1.0 - ed)
    m_rf = r_f_prev * ef + theta_f * (1.0 - ef)
    m_Id = (r_d_prev - theta_d) * n_d + theta_d * tau
    m_If = (r_f_prev - theta_f) * n_f + theta_f * tau

    return np.stack([m_rd, m_rf, m_Id, m_If], axis=1)


# ---------------------------------------------------------------------------
# Futures expectations under Q_d
# ---------------------------------------------------------------------------

def futures_f(r_f_t: float, t: float, U: float, T: float, p: ModelParams) -> float:
    """
    F^f_t(U, T): expectation of foreign simple rate R^f(U, T) under the
    *foreign* risk-neutral measure Q̂  (paper notation E_{Q̂}[R^f(U,T)|F_t]).

    Under Q̂, r^f follows:
        dr^f = (â − b̂ r^f) dt + σ̂ dŴ^2
    with â = ĉ + σ̂·σ̄·ρ₂₃  (Girsanov drift shift from Q_d to Q̂).

    Using: 1 + δ·R^f(U,T) = exp(∫_U^T r^f_u du) under Q̂ dynamics.
    E^{Q̂}[exp(I^f_{U,T}) | F_t] = exp(m + v/2).

    The Γ correction in the pricing formulas accounts for the difference
    between Q̂ and Q_d expectations (via the ρ₂₃ term in eq. 4.9).

    Parameters
    ----------
    r_f_t : float  Foreign rate at time t (t <= U).
    t     : float  Current time.
    U     : float  Accrual start.
    T     : float  Accrual end.
    p     : ModelParams

    Returns
    -------
    float  F^f_t(U, T)  (in decimal), under Q̂.
    """
    delta = T - U
    # Drift under foreign risk-neutral measure Q̂:  â = ĉ + σ̂·σ̄·ρ₂₃
    a_hat = p.c_hat + p.sigma_hat * p.sigma_bar * p.rho_23
    theta_f_hat = a_hat / p.b_hat          # long-run mean under Q̂

    # --- Conditional distribution of r^f_U given F_t under Q̂ ---
    tau_tU = U - t
    m_rfU = vasicek_mean(r_f_t, tau_tU, theta_f_hat, p.b_hat)
    v_rfU = vasicek_var_r(tau_tU, p.b_hat, p.sigma_hat)

    # --- Conditional distribution of I^f_{U,T} given r^f_U under Q̂ ---
    n_UT = n_func(T - U, p.b_hat)
    var_cond_UT = integrated_rate_var(T - U, p.b_hat, p.sigma_hat)

    # Law of total expectation / variance (under Q̂):
    m_If_UT = n_UT * m_rfU + theta_f_hat * (delta - n_UT)
    v_If_UT = var_cond_UT + n_UT ** 2 * v_rfU

    E_exp = np.exp(m_If_UT + 0.5 * v_If_UT)
    return (E_exp - 1.0) / delta


def futures_d(r_d_t: float, t: float, U: float, T: float, p: ModelParams) -> float:
    """
    F^d_t(U, T): expectation of domestic simple rate R^d(U, T) under Q_d.

    Analogous to futures_f but for the domestic factor.
    """
    delta = T - U
    theta_d = p.a / p.b

    tau_tU = U - t
    m_rdU = vasicek_mean(r_d_t, tau_tU, theta_d, p.b)
    v_rdU = vasicek_var_r(tau_tU, p.b, p.sigma)

    n_UT = n_func(T - U, p.b)
    var_cond_UT = integrated_rate_var(T - U, p.b, p.sigma)

    m_Id_UT = n_UT * m_rdU + theta_d * (delta - n_UT)
    v_Id_UT = var_cond_UT + n_UT ** 2 * v_rdU

    E_exp = np.exp(m_Id_UT + 0.5 * v_Id_UT)
    return (E_exp - 1.0) / delta


# ---------------------------------------------------------------------------
# Extended covariance for FX simulation (for exp_4)
# State vector: (r^d, r^f, log Q, I^d, I^f) over one step
# ---------------------------------------------------------------------------

def step_cov_matrix_5d(tau: float, p: ModelParams) -> np.ndarray:
    """
    5×5 conditional covariance for (r^d, r^f, log Q increment, I^d, I^f).

    Indexing: 0=r^d, 1=r^f, 2=Delta log Q, 3=I^d, 4=I^f.

    log Q_t follows Gaussian dynamics; we track the increment Delta log Q over [0,tau].

    Parameters
    ----------
    tau : float
    p   : ModelParams

    Returns
    -------
    np.ndarray  Shape (5,5).
    """
    C = np.zeros((5, 5))

    # Sub-block (0,1,3,4): same as 4×4 for (r^d, r^f, I^d, I^f)
    base = step_cov_matrix(tau, p)
    idx = [0, 1, 3, 4]
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            C[ii, jj] = base[i, j]

    # Delta log Q = integral of (r^d - r^f + lambda^q) dt  + sigma_bar * (Z^3 increment)
    #             = (deterministic) + sigma_bar * sqrt(tau) * eps3
    # But more precisely, its diffusion part:
    # Cov[Delta log Q, r^d] = rho_13 * sigma * sigma_bar * ∫_0^tau e^{-b(tau-u)} du
    #                        = rho_13 * sigma * sigma_bar * n(tau; b)
    n_d = n_func(tau, p.b)
    n_f = n_func(tau, p.b_hat)

    C[0, 2] = C[2, 0] = p.rho_13 * p.sigma * p.sigma_bar * n_d
    C[1, 2] = C[2, 1] = p.rho_23 * p.sigma_hat * p.sigma_bar * n_f
    C[3, 2] = C[2, 3] = p.rho_13 * p.sigma * p.sigma_bar / p.b * (tau - n_d)
    C[4, 2] = C[2, 4] = p.rho_23 * p.sigma_hat * p.sigma_bar / p.b_hat * (tau - n_f)
    C[2, 2] = p.sigma_bar ** 2 * tau

    return C
