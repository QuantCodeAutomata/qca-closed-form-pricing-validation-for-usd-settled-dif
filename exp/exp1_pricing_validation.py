"""
Experiment 1: Closed-form pricing validation for USD-settled differential swaps.

Compares analytical prices (Propositions 4.1 and 4.2) with Monte Carlo
estimates under the domestic risk-neutral measure using exact Gaussian simulation.

Table 2-style results: analytical price, MC price, error, SE, z-score.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.params import ModelParams, ContractParams
from src.schedule import make_schedule
from src.differential_swap import (
    single_period_value, multi_period_value, par_margin
)
from src.monte_carlo import (
    mc_single_period, mc_multi_period, mc_validation_table
)


# ---------------------------------------------------------------------------
# Baseline model parameters (Table 1)
# ---------------------------------------------------------------------------

P = ModelParams()   # all defaults match Table 1


# ---------------------------------------------------------------------------
# Helper: single-period analytical price at t=0
# ---------------------------------------------------------------------------

def analytic_single(U: float, T: float, gamma: int, collateral: str,
                    kappa: float | None = None) -> float:
    """
    Compute the analytical single-period value at t=0.

    If kappa is None, sets kappa to the par margin first.
    """
    if kappa is None:
        # Par margin: set value to 0
        kappa = _par_single(U, T, gamma, collateral)
    c = ContractParams(gamma=gamma, kappa=kappa, beta=1.0, collateral=collateral)
    return single_period_value(0.0, U, T, P.r_d_0, P.r_f_0, c, P)


def _par_single(U: float, T: float, gamma: int, collateral: str) -> float:
    """
    Par margin for a single-period contract (kappa that makes value = 0).

    Solve: I^(1) - gamma*I^(2) - (1-gamma+delta*kappa)*I^(3) = 0
    => kappa = [I^(1) - gamma*I^(2) - (1-gamma)*I^(3)] / [delta * I^(3)]
    """
    from src.differential_swap import coupon_I1, coupon_I2, coupon_I3, collateral_multiplier
    c_tmp = ContractParams(gamma=gamma, kappa=0.0, beta=1.0, collateral=collateral)
    I1 = coupon_I1(0.0, U, T, P.r_d_0, 1.0, collateral, P)
    I2 = coupon_I2(0.0, U, T, P.r_d_0, P.r_f_0, 1.0, collateral, P)
    I3 = coupon_I3(0.0, T, P.r_d_0, 1.0, collateral, P)
    delta = T - U
    return (I1 - gamma * I2 - (1.0 - gamma) * I3) / (delta * I3)


# ---------------------------------------------------------------------------
# Validation cases (Table 2 style)
# ---------------------------------------------------------------------------

def run_validation(n_paths: int = 250_000, seed: int = 42) -> pd.DataFrame:
    """
    Run all four validation cases and return results as a DataFrame.

    Cases:
      1. Single-period T=0.5, U=0, gamma=0, USD collateral
      2. Single-period T=1.5, U=1.0, gamma=1, EUR collateral
      3. Multi-period 3Y, semi-annual, gamma=0, USD collateral
      4. Multi-period 3Y, semi-annual, gamma=1, EUR collateral
    """
    records = []

    # ---- Case 1: Single-period, T=0.5, U=0, gamma=0, USD ----
    U1, T1, g1, coll1 = 0.0, 0.5, 0, "USD"
    kappa1 = _par_single(U1, T1, g1, coll1)
    c1 = ContractParams(gamma=g1, kappa=kappa1, beta=1.0, collateral=coll1)
    analytic1 = single_period_value(0.0, U1, T1, P.r_d_0, P.r_f_0, c1, P)
    mc1, se1 = mc_single_period(0.0, U1, T1, c1, P, n_paths=n_paths, seed=seed)
    records.append(mc_validation_table(
        analytic1, mc1, se1,
        label="SP T=0.5 U=0.0 gamma=0 USD"
    ))

    # ---- Case 2: Single-period, T=1.5, U=1.0, gamma=1, EUR ----
    U2, T2, g2, coll2 = 1.0, 1.5, 1, "EUR"
    kappa2 = _par_single(U2, T2, g2, coll2)
    c2 = ContractParams(gamma=g2, kappa=kappa2, beta=1.0, collateral=coll2)
    analytic2 = single_period_value(0.0, U2, T2, P.r_d_0, P.r_f_0, c2, P)
    mc2, se2 = mc_single_period(0.0, U2, T2, c2, P, n_paths=n_paths, seed=seed)
    records.append(mc_validation_table(
        analytic2, mc2, se2,
        label="SP T=1.5 U=1.0 gamma=1 EUR"
    ))

    # ---- Cases 3 & 4: Multi-period, 3Y, semi-annual ----
    dates3, deltas3 = make_schedule(maturity=3.0, freq=0.5)

    for g3, coll3 in [(0, "USD"), (1, "EUR")]:
        kappa3 = par_margin(
            0.0, dates3, deltas3, P.r_d_0, P.r_f_0,
            g3, 1.0, coll3, P
        )
        c3 = ContractParams(gamma=g3, kappa=kappa3, beta=1.0, collateral=coll3)
        analytic3 = multi_period_value(
            0.0, dates3, deltas3, P.r_d_0, P.r_f_0, c3, P
        )
        mc3, se3 = mc_multi_period(dates3, deltas3, c3, P, n_paths=n_paths, seed=seed)
        label = f"MP 3Y semi-annual gamma={g3} {coll3}"
        records.append(mc_validation_table(analytic3, mc3, se3, label=label))

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------------

def check_internal_consistency() -> dict:
    """
    Verify:
    (a) Par margin invariance to collateral currency when net carries equal.
    (b) Foreign-parameter invariance when gamma=0.
    (c) Par margin for single-period at-par gives zero value.
    """
    results = {}

    dates, deltas = make_schedule(3.0, 0.5)

    # (a) When carry_f = carry_d = 0, USD and EUR collateral give the same par margin
    from copy import copy
    P_equal = copy(P)
    P_equal.carry_f = P_equal.carry_d   # make carries equal
    kappa_usd = par_margin(0.0, dates, deltas, P_equal.r_d_0, P_equal.r_f_0,
                           1, 1.0, "USD", P_equal)
    kappa_eur = par_margin(0.0, dates, deltas, P_equal.r_d_0, P_equal.r_f_0,
                           1, 1.0, "EUR", P_equal)
    results["equal_carry_par_diff"] = abs(kappa_usd - kappa_eur)

    # (b) gamma=0: foreign FX param change should not affect price
    kappa_g0 = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                          0, 1.0, "USD", P)
    P_alt_fx = copy(P)
    P_alt_fx.sigma_bar = 0.15   # 2x FX vol
    P_alt_fx.rho_23 = -0.30
    kappa_g0_alt = par_margin(0.0, dates, deltas, P_alt_fx.r_d_0, P_alt_fx.r_f_0,
                              0, 1.0, "USD", P_alt_fx)
    results["gamma0_fx_invariance"] = abs(kappa_g0 - kappa_g0_alt)

    # (c) At-par value should be ~ 0
    kappa_par = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                           1, 1.0, "USD", P)
    c_par = ContractParams(gamma=1, kappa=kappa_par, beta=1.0, collateral="USD")
    v_at_par = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0, c_par, P)
    results["at_par_value_near_zero"] = abs(v_at_par)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 1: Closed-form Pricing Validation")
    print("=" * 70)

    # Run validation
    df = run_validation(n_paths=250_000, seed=42)

    print("\nTable 2-style validation results:")
    print("-" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    # Internal consistency
    checks = check_internal_consistency()
    print("\nInternal consistency checks:")
    for k, v in checks.items():
        print(f"  {k}: {v:.2e}")

    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/exp1_table2.csv", index=False)

    # Plot z-scores
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(df["label"], df["z_score"], color=["steelblue" if abs(z) < 2 else "firebrick"
                                                for z in df["z_score"]])
    ax.axvline(-2, color="red", linestyle="--", linewidth=0.8, label="±2 SE")
    ax.axvline(+2, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("z-score  (MC - Analytic) / SE")
    ax.set_title("Exp 1 — MC vs Analytical Validation: z-scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/exp1_zscores.png", dpi=150)
    plt.close()

    print("\nResults saved to results/exp1_table2.csv and results/exp1_zscores.png")
    return df, checks


if __name__ == "__main__":
    main()
