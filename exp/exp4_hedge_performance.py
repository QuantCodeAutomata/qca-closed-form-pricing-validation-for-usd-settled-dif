"""
Experiment 4: Discrete Rebalancing Performance of SOFR and €STR Futures Hedges.

Simulates daily Euler-Maruyama paths and computes terminal replication errors
for unhedged, SOFR-only, and full SOFR+€STR hedge strategies.

Reproduces:
  - Table 4: RMSE and variance ratio for daily rebalancing
  - Figure 3: RMSE vs rebalancing frequency (EUR collateral, gamma=1)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copy import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.params import ModelParams, ContractParams
from src.schedule import make_schedule
from src.differential_swap import par_margin, multi_period_value
from src.vasicek import (
    bond_d, futures_d, futures_f, n_func,
    vasicek_mean, vasicek_var_r, integrated_rate_mean, integrated_rate_var
)
from src.quanto import gamma_correction
from src.differential_swap import coupon_I1, coupon_I2, coupon_I3, collateral_multiplier


P = ModelParams()


# ---------------------------------------------------------------------------
# Vectorised futures-level functions
# ---------------------------------------------------------------------------

def futures_d_vec(r_d: np.ndarray, t: float, U: float, T: float,
                  p: ModelParams,
                  I_d_accrued: np.ndarray | None = None) -> np.ndarray:
    """
    F^d_t(U,T) for each path (vectorised).

    For t <= U: standard conditional expectation via r^d_U distribution.
    For t > U:  I^d_{U,T} = I^d_{U,t} (known, = I_d_accrued) + I^d_{t,T} (random).
                E[exp(I^d_{U,T})|F_t] = exp(I_d_accrued) * exp(m_{t,T} + v_{t,T}/2).

    Parameters
    ----------
    I_d_accrued : np.ndarray | None
        Accumulated ∫_U^t r^d_u du for each path. Required when t > U.
    """
    delta = T - U
    theta_d = p.a / p.b

    if t <= U + 1e-12:
        # t <= U: propagate r^d distribution from t to U, then from U to T
        tau_tU = U - t
        m_rdU = r_d * np.exp(-p.b * tau_tU) + theta_d * (1.0 - np.exp(-p.b * tau_tU))
        v_rdU = vasicek_var_r(tau_tU, p.b, p.sigma)
        n_UT = n_func(delta, p.b)
        var_cond = integrated_rate_var(delta, p.b, p.sigma)
        m_Id_UT = n_UT * m_rdU + theta_d * (delta - n_UT)
        v_Id_UT = var_cond + n_UT ** 2 * v_rdU
        E_exp = np.exp(m_Id_UT + 0.5 * v_Id_UT)
    else:
        # t > U: decompose I^d_{U,T} = I^d_{U,t} + I^d_{t,T}
        tau_tT = T - t
        m_Id_tT = integrated_rate_mean(r_d, tau_tT, theta_d, p.b)
        v_Id_tT = integrated_rate_var(tau_tT, p.b, p.sigma)
        I_Ut = I_d_accrued if I_d_accrued is not None else np.zeros_like(r_d)
        E_exp = np.exp(I_Ut) * np.exp(m_Id_tT + 0.5 * v_Id_tT)

    return (E_exp - 1.0) / delta


def futures_f_vec(r_f: np.ndarray, t: float, U: float, T: float,
                  p: ModelParams,
                  I_f_accrued: np.ndarray | None = None) -> np.ndarray:
    """
    F^f_t(U,T) for each path (vectorised).

    For t > U: I^f_{U,T} = I^f_{U,t} + I^f_{t,T}, split at known/random boundary.

    Parameters
    ----------
    I_f_accrued : np.ndarray | None
        Accumulated ∫_U^t r^f_u du. Required when t > U.
    """
    delta = T - U
    theta_f = p.c_hat / p.b_hat

    if t <= U + 1e-12:
        tau_tU = U - t
        m_rfU = r_f * np.exp(-p.b_hat * tau_tU) + theta_f * (1.0 - np.exp(-p.b_hat * tau_tU))
        v_rfU = vasicek_var_r(tau_tU, p.b_hat, p.sigma_hat)
        n_UT = n_func(delta, p.b_hat)
        var_cond = integrated_rate_var(delta, p.b_hat, p.sigma_hat)
        m_If_UT = n_UT * m_rfU + theta_f * (delta - n_UT)
        v_If_UT = var_cond + n_UT ** 2 * v_rfU
        E_exp = np.exp(m_If_UT + 0.5 * v_If_UT)
    else:
        tau_tT = T - t
        m_If_tT = integrated_rate_mean(r_f, tau_tT, theta_f, p.b_hat)
        v_If_tT = integrated_rate_var(tau_tT, p.b_hat, p.sigma_hat)
        I_Ut = I_f_accrued if I_f_accrued is not None else np.zeros_like(r_f)
        E_exp = np.exp(I_Ut) * np.exp(m_If_tT + 0.5 * v_If_tT)

    return (E_exp - 1.0) / delta


# ---------------------------------------------------------------------------
# Vectorised hedge ratios
# ---------------------------------------------------------------------------

def phi_d_vec(t: float, U_j: float, T_j: float, delta_j: float,
              r_d: np.ndarray, r_f: np.ndarray,
              c: ContractParams, p: ModelParams,
              I_d_accrued: np.ndarray | None = None,
              I_f_accrued: np.ndarray | None = None) -> np.ndarray:
    """
    Domestic (SOFR) hedge ratio phi^{d,j}_t for all paths.

    For t <= U_j:
      phi = [sigma_Uj(t)*I1 - sigma_Tj(t)*(gamma*I2 + (1-gamma+delta*kappa)*I3)]
             / (sigma_Tj(t) * B^d(t, T_j))
    For t in (U_j, T_j]:
      phi = delta_j * (gamma*I2 + (1-gamma+delta*kappa)*I3) / (1 + delta_j*F^d_t)

    Parameters
    ----------
    I_d_accrued : optional array  ∫_{U_j}^t r^d du (needed in t > U_j branch).
    I_f_accrued : optional array  ∫_{U_j}^t r^f du (needed for F^f in t > U_j branch).
    """
    alpha_val = p.alpha_beta(c.beta, c.collateral)
    A = np.exp(-alpha_val * (T_j - t))

    # Vectorised bond prices — bond_d works with ndarray r_d via numpy ops
    bd_Tj = bond_d(r_d, T_j - t, p)
    Ff = futures_f_vec(r_f, t, U_j, T_j, p, I_f_accrued=I_f_accrued)
    G = gamma_correction(t, U_j, T_j, p)

    I2 = A * bd_Tj * (1.0 + delta_j * Ff) * G
    I3 = A * bd_Tj

    if t <= U_j + 1e-12:
        # Vasicek loadings (both ≤ 0; sigma_T is more negative than sigma_U since T > U)
        sigma_Tj = -p.sigma * n_func(T_j - t, p.b)
        sigma_Uj = -p.sigma * n_func(U_j - t, p.b)
        bd_Uj = bond_d(r_d, max(U_j - t, 0.0), p)
        I1 = A * bd_Uj

        num = sigma_Uj * I1 - sigma_Tj * (c.gamma * I2 + (1.0 - c.gamma + delta_j * c.kappa) * I3)

        # Correct denominator = ν^{d,j}_t = (1 + δ F^d_t) * (σ_Uj − σ_Tj) / δ_j
        # Derivation: ∂F^d/∂r^d · σ = (1+δF^d) · n(T−t,b)−n(U−t,b) = (1+δF^d)·(σ_U−σ_T)/δ
        # This is always positive (σ_U − σ_T > 0 because T > U → |σ_T| > |σ_U|).
        Fd = futures_d_vec(r_d, t, U_j, T_j, p, I_d_accrued=I_d_accrued)
        denom = (1.0 + delta_j * Fd) * (sigma_Uj - sigma_Tj) / delta_j
        return np.where(np.abs(denom) > 1e-15, num / denom, 0.0)
    else:
        # Ex-accrual branch (simplified Prop 5.2 formula for t ∈ [T_{j-1}, T_j)):
        #   phi = δ_j · (γ I_j^(2) + (1−γ+δκ) I_j^(3)) / (1 + δ_j F^d_t)
        Fd = futures_d_vec(r_d, t, U_j, T_j, p, I_d_accrued=I_d_accrued)
        denom = 1.0 + delta_j * Fd
        return delta_j * (c.gamma * I2 + (1.0 - c.gamma + delta_j * c.kappa) * I3) / denom


def phi_f_vec(t: float, U_j: float, T_j: float, delta_j: float,
              r_d: np.ndarray, r_f: np.ndarray, Q: np.ndarray,
              c: ContractParams, p: ModelParams) -> np.ndarray:
    """
    Foreign (€STR) hedge ratio phi^{f,j}_t = -(gamma*delta_j / Q_t) * A * B^d(t,T_j) * Gamma.
    """
    if c.gamma == 0:
        return np.zeros(len(r_d))

    alpha_val = p.alpha_beta(c.beta, c.collateral)
    A = np.exp(-alpha_val * (T_j - t))
    # Vectorised — bond_d supports ndarray r_d
    bd_Tj = bond_d(r_d, T_j - t, p)
    G = gamma_correction(t, U_j, T_j, p)

    return -(c.gamma * delta_j / Q) * A * bd_Tj * G


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_terminal_errors(
    coupon_dates: np.ndarray,
    deltas: np.ndarray,
    c: ContractParams,
    p: ModelParams,
    n_steps_per_period: int = 126,   # 126 = daily, 26 ≈ weekly, 6 ≈ monthly
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Simulate terminal replication errors for three hedge strategies.

    Parameters
    ----------
    coupon_dates        : np.ndarray  T_0..T_n.
    deltas              : np.ndarray  Accrual fractions.
    c                   : ContractParams  kappa = USD-par.
    p                   : ModelParams
    n_steps_per_period  : int  Steps per 0.5Y coupon period.
    n_paths             : int
    seed                : int

    Returns
    -------
    dict with 'unhedged', 'sofr', 'full', each shape (n_paths,).
    Errors in decimal (multiply by 1e4 for bp).
    """
    n_coupons = len(deltas)
    dt = deltas[0] / n_steps_per_period   # assumes constant accrual

    rng = np.random.default_rng(seed)
    corr_mat = np.array([
        [1.0,      p.rho_12, p.rho_13],
        [p.rho_12, 1.0,      p.rho_23],
        [p.rho_13, p.rho_23, 1.0     ],
    ])
    assert np.all(np.linalg.eigvalsh(corr_mat) > -1e-12), "Corr matrix not PSD"
    L = np.linalg.cholesky(corr_mat + 1e-10 * np.eye(3))

    alpha = p.alpha_beta(c.beta, c.collateral)

    # State variables (all paths simultaneously)
    r_d = np.full(n_paths, p.r_d_0)
    r_f = np.full(n_paths, p.r_f_0)
    Q   = np.full(n_paths, p.Q_0)
    cum_I_d = np.zeros(n_paths)   # cumulative ∫_0^t r^d du

    # Within-coupon integrals (for payoff realisation)
    I_d_coupon = np.zeros(n_paths)
    I_f_coupon = np.zeros(n_paths)

    # Futures P&L accumulators — PRESENT-VALUE (PV) basis.
    # Each futures gain at t_{k+1} is multiplied by the path-realised discount factor
    # D(t_{k+1}) = exp(−α t_{k+1} − ∫_0^{t_{k+1}} r^d_u du) so it is comparable to
    # ideal_pv = Σ_j exp(−α T_j − ∫_0^{T_j} r^d) · payoff_j (also in PV terms).
    # This is equivalent to the terminal-dollar approach after dividing by exp(∫_0^{T_n} r^d).
    pnl_sofr = np.zeros(n_paths)
    pnl_full = np.zeros(n_paths)

    # Present-value sum of discounted coupon payoffs (ideal replication target)
    ideal_pv = np.zeros(n_paths)

    # Initial futures levels for each coupon
    Fd_prev = {}
    Ff_prev = {}
    for j in range(n_coupons):
        Uj = coupon_dates[j]
        Tj = coupon_dates[j + 1]
        Fd_prev[j] = futures_f_vec(np.full(n_paths, p.r_d_0), 0.0, Uj, Tj, p) * 0  # placeholder
        Fd_prev[j] = np.array([futures_d(p.r_d_0, 0.0, Uj, Tj, p)] * n_paths)
        Ff_prev[j] = np.array([futures_f(p.r_f_0, 0.0, Uj, Tj, p)] * n_paths)

    current_coupon = 0   # index into coupon_dates

    for j_coupon in range(n_coupons):
        Uj = coupon_dates[j_coupon]
        Tj = coupon_dates[j_coupon + 1]
        delta_j = deltas[j_coupon]
        dt = delta_j / n_steps_per_period

        I_d_coupon[:] = 0.0
        I_f_coupon[:] = 0.0

        for k in range(n_steps_per_period):
            t_gl   = coupon_dates[j_coupon] + k * dt
            t_next = t_gl + dt

            sqrt_dt = np.sqrt(dt)
            eps = rng.standard_normal((n_paths, 3)) @ L.T

            # ---- Save pre-Euler state (used for phi at t_gl) ----
            r_d_pre = r_d.copy()
            r_f_pre = r_f.copy()
            Q_pre   = Q.copy()

            # Left-endpoint accumulation (uses pre-update rates)
            I_d_coupon += r_d_pre * dt
            I_f_coupon += r_f_pre * dt
            cum_I_d    += r_d_pre * dt

            # Euler update → new state at t_next
            r_d = r_d_pre + (p.a - p.b * r_d_pre) * dt + p.sigma * sqrt_dt * eps[:, 0]
            r_f = r_f_pre + (p.c_hat - p.b_hat * r_f_pre) * dt + p.sigma_hat * sqrt_dt * eps[:, 1]
            Q   = np.maximum(
                Q_pre * np.exp(
                    (r_d_pre - r_f_pre) * dt + p.sigma_bar * sqrt_dt * eps[:, 2]
                    - 0.5 * p.sigma_bar ** 2 * dt
                ),
                1e-10,
            )

            # -- Futures P&L for all outstanding coupons --
            for jj in range(j_coupon, n_coupons):
                Ujj      = coupon_dates[jj]
                Tjj      = coupon_dates[jj + 1]
                delta_jj = deltas[jj]

                # During current coupon (jj == j_coupon): t_next may be > Ujj
                # → pass accumulated integrals so futures use correct decomposition
                is_current = (jj == j_coupon)
                Idc = I_d_coupon if (is_current and t_next > Ujj + 1e-12) else None
                Ifc = I_f_coupon if (is_current and t_next > Ujj + 1e-12) else None

                # Futures at t_next (post-update state) for P&L calculation
                curr_Fd = futures_d_vec(r_d, t_next, Ujj, Tjj, p, I_d_accrued=Idc)
                curr_Ff = futures_f_vec(r_f, t_next, Ujj, Tjj, p, I_f_accrued=Ifc)

                Delta_Fd = curr_Fd - Fd_prev[jj]
                Delta_Ff = curr_Ff - Ff_prev[jj]

                # Hedge ratios at t_gl using PRE-update state
                Idc_pre = I_d_coupon - r_d_pre * dt if (is_current and t_gl > Ujj + 1e-12) else None
                Ifc_pre = I_f_coupon - r_f_pre * dt if (is_current and t_gl > Ujj + 1e-12) else None
                phi_d = phi_d_vec(t_gl, Ujj, Tjj, delta_jj,
                                  r_d_pre, r_f_pre, c, p,
                                  I_d_accrued=Idc_pre, I_f_accrued=Ifc_pre)

                # Discount factor at t_next — brings each gain into t=0 PV.
                # D(t) = exp(−α_eff · t − ∫_0^t r^d_u du), constant spread α_eff.
                D_eff = np.exp(-alpha * t_next - cum_I_d)

                pnl_sofr += D_eff * phi_d * Delta_Fd

                if c.gamma > 0:
                    phi_f = phi_f_vec(t_gl, Ujj, Tjj, delta_jj,
                                      r_d_pre, r_f_pre, Q_pre, c, p)
                    # Right-end FX conversion per paper: Q_{t_{k+1}} * ΔF^f
                    pnl_full += D_eff * (phi_d * Delta_Fd + phi_f * Q * Delta_Ff)
                else:
                    pnl_full += D_eff * phi_d * Delta_Fd

                Fd_prev[jj] = curr_Fd
                Ff_prev[jj] = curr_Ff

        # --- Coupon payment at T_j ---
        payoff_j = (
            np.exp(I_d_coupon)
            - c.gamma * np.exp(I_f_coupon)
            - (1.0 - c.gamma + delta_j * c.kappa)
        )
        # Discounted coupon PV: A_{0,T_j}^{β,i} · exp(−∫_0^{T_j} r^d) · payoff_j
        A_j = np.exp(-alpha * Tj)
        D_j = np.exp(-cum_I_d)   # path-realised B^d(0, T_j)
        ideal_pv += A_j * D_j * payoff_j

    # Replication errors in PV terms.
    # Unhedged: 0 futures P&L → error = −ideal_pv.
    # Perfect hedge: pnl → ideal_pv → error ≈ 0.
    return {
        "unhedged": -ideal_pv,
        "sofr":     pnl_sofr - ideal_pv,
        "full":     pnl_full - ideal_pv,
    }


def compute_metrics(errors: dict, notional: float = 1.0) -> pd.DataFrame:
    """
    Compute RMSE (in bp of notional) and variance ratio vs unhedged.

    Parameters
    ----------
    errors   : dict  {'unhedged': ..., 'sofr': ..., 'full': ...}
    notional : float  Swap notional.

    Returns
    -------
    pd.DataFrame
    """
    var_unhd = np.var(errors["unhedged"])
    records = []
    for strategy in ["unhedged", "sofr", "full"]:
        err = errors[strategy]
        rmse_bp = 1e4 * np.sqrt(np.mean(err ** 2)) / notional
        var_ratio = np.var(err) / var_unhd if var_unhd > 0 else 0.0
        records.append({
            "strategy": strategy,
            "rmse_bp": rmse_bp,
            "var_ratio": var_ratio,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table 4 and Figure 3
# ---------------------------------------------------------------------------

def run_table4(n_paths: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Table 4: Daily rebalancing RMSE and variance ratio.
    Cases: gamma x collateral x strategy.
    """
    dates, deltas = make_schedule(3.0, freq=0.5)
    kappa_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                           1, 1.0, "USD", P)   # gamma=1 par for reference

    records = []
    for gamma_val in [0, 1]:
        # USD-collateral par for this gamma
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                           gamma_val, 1.0, "USD", P)
        for coll in ["USD", "EUR"]:
            c = ContractParams(gamma=gamma_val, kappa=k_usd,
                               beta=1.0, collateral=coll)
            errors = simulate_terminal_errors(
                dates, deltas, c, P,
                n_steps_per_period=126,   # daily
                n_paths=n_paths, seed=seed,
            )
            metrics = compute_metrics(errors)
            for _, row in metrics.iterrows():
                records.append({
                    "gamma": gamma_val,
                    "collateral": coll,
                    "strategy": row["strategy"],
                    "rmse_bp": row["rmse_bp"],
                    "var_ratio": row["var_ratio"],
                })

    return pd.DataFrame(records)


def run_figure3(n_paths: int = 10_000, seed: int = 42) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Figure 3: RMSE vs rebalancing frequency for EUR collateral, gamma=1.

    Frequencies: daily (126 steps), weekly (~26 steps), monthly (~6 steps).
    """
    dates, deltas = make_schedule(3.0, freq=0.5)
    k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 1, 1.0, "USD", P)
    c = ContractParams(gamma=1, kappa=k_usd, beta=1.0, collateral="EUR")

    freq_map = {
        "daily": 126,
        "weekly": 26,
        "monthly": 6,
    }

    records = []
    for freq_label, n_steps in freq_map.items():
        errors = simulate_terminal_errors(
            dates, deltas, c, P,
            n_steps_per_period=n_steps,
            n_paths=n_paths, seed=seed,
        )
        metrics = compute_metrics(errors)
        for _, row in metrics.iterrows():
            records.append({
                "freq": freq_label,
                "n_steps": n_steps,
                "strategy": row["strategy"],
                "rmse_bp": row["rmse_bp"],
            })

    df = pd.DataFrame(records)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"unhedged": "grey", "sofr": "firebrick", "full": "steelblue"}
    for strat in ["unhedged", "sofr", "full"]:
        sub = df[df.strategy == strat].sort_values("n_steps", ascending=False)
        ax.plot(sub["freq"], sub["rmse_bp"],
                marker="o", label=strat, color=colors[strat])
    ax.set_xlabel("Rebalancing frequency")
    ax.set_ylabel("RMSE (bp of notional)")
    ax.set_title("Figure 3 — Hedge RMSE vs Rebalancing (EUR collateral, gamma=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return df, fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 4: Discrete Rebalancing Hedge Performance")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    n_paths = 10_000
    seed = 42

    print(f"\nRunning Table 4 ({n_paths} paths, daily rebalancing)...")
    table4 = run_table4(n_paths=n_paths, seed=seed)
    print("\nTable 4:")
    print(table4.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    table4.to_csv("results/exp4_table4.csv", index=False)

    print(f"\nRunning Figure 3 ({n_paths} paths, 3 rebalancing frequencies)...")
    fig3_data, fig3 = run_figure3(n_paths=n_paths, seed=seed)
    fig3.savefig("results/exp4_figure3.png", dpi=150)
    plt.close(fig3)
    fig3_data.to_csv("results/exp4_figure3_data.csv", index=False)
    print("Figure 3 saved to results/exp4_figure3.png")

    print("\nResults saved: results/exp4_table4.csv, results/exp4_figure3.png")
    return table4, fig3_data


if __name__ == "__main__":
    main()
