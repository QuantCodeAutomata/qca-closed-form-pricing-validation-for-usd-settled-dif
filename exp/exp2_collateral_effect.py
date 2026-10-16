"""
Experiment 2: Collateral Currency Valuation Effect.

Quantifies how collateral currency (USD vs EUR) changes the value and par margin
of the same USD-settled swap under full collateralization.

Reproduces:
  - Table 3: par-margin change and CSA-switch value for 5Y and 10Y
  - Figure 1: EUR-minus-USD value adjustment vs fixed margin grid
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.params import ModelParams, ContractParams
from src.schedule import make_schedule
from src.differential_swap import multi_period_value, par_margin


P = ModelParams()   # baseline Table 1 parameters
NOTIONAL = 100_000_000.0   # 100M USD for value reporting


def compute_par_margins(maturities: list[float],
                        gammas: list[int]) -> pd.DataFrame:
    """
    Compute USD-collateral and EUR-collateral par margins for each maturity/gamma.

    Returns DataFrame with columns: maturity, gamma, kappa_USD, kappa_EUR,
    par_change_bp (EUR-USD in bp).
    """
    records = []
    for mat in maturities:
        dates, deltas = make_schedule(mat, freq=0.5)
        for g in gammas:
            k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "USD", P)
            k_eur = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "EUR", P)
            records.append({
                "maturity": mat,
                "gamma": g,
                "kappa_USD": k_usd,
                "kappa_EUR": k_eur,
                "par_change_bp": (k_eur - k_usd) * 1e4,
            })
    return pd.DataFrame(records)


def compute_collateral_value_adjustment(
    mat: float, gamma: int, kappa: float
) -> float:
    """
    Delta V_0^{f-d}(kappa) = DS_0(EUR; kappa) - DS_0(USD; kappa)

    in unit-notional terms (multiply by NOTIONAL for USD value).
    """
    dates, deltas = make_schedule(mat, freq=0.5)
    c_eur = ContractParams(gamma=gamma, kappa=kappa, beta=1.0, collateral="EUR")
    c_usd = ContractParams(gamma=gamma, kappa=kappa, beta=1.0, collateral="USD")
    v_eur = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0, c_eur, P)
    v_usd = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0, c_usd, P)
    return v_eur - v_usd


def build_table3() -> pd.DataFrame:
    """
    Reproduce Table 3: fixed margin = USD-par + 100bp, maturities 5Y and 10Y.

    Returns DataFrame with par-margin change (bp) and CSA-switch value (USD).
    """
    maturities = [5.0, 10.0]
    gammas = [0, 1]
    records = []
    for mat in maturities:
        dates, deltas = make_schedule(mat, freq=0.5)
        for g in gammas:
            # USD-collateral par margin
            k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "USD", P)
            k_eur = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "EUR", P)
            # Fixed margin: USD par + 100bp
            kappa_fixed = k_usd + 0.0100

            delta_v = compute_collateral_value_adjustment(mat, g, kappa_fixed)
            csa_switch_value = NOTIONAL * delta_v

            records.append({
                "maturity_Y": mat,
                "gamma": g,
                "kappa_USD_par": k_usd,
                "kappa_EUR_par": k_eur,
                "par_change_bp": (k_eur - k_usd) * 1e4,
                "kappa_fixed": kappa_fixed,
                "CSA_switch_USD": csa_switch_value,
            })
    return pd.DataFrame(records)


def build_figure1(
    maturities: list[float] = [1.0, 3.0, 5.0, 10.0],
    gamma: int = 0,
    n_grid: int = 81,
    range_bp: float = 200.0,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Reproduce Figure 1: EUR-minus-USD value vs fixed margin for multiple maturities.

    x-axis: (kappa - kappa_USD_par) in bp  [-200, +200]
    y-axis: NOTIONAL * Delta_V^{f-d}(kappa)  in USD
    """
    x_grid = np.linspace(-range_bp, range_bp, n_grid)   # in bp
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    all_records = []
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(maturities)))

    for ax, g in zip(axes, [0, 1]):
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        for mat, col in zip(maturities, colors):
            dates, deltas = make_schedule(mat, freq=0.5)
            k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "USD", P)
            y_vals = []
            for x in x_grid:
                kappa = k_usd + x / 1e4
                dv = compute_collateral_value_adjustment(mat, g, kappa)
                y_vals.append(NOTIONAL * dv)
                all_records.append({
                    "maturity": mat,
                    "gamma": g,
                    "x_bp": x,
                    "kappa": kappa,
                    "delta_V_USD": NOTIONAL * dv,
                })
            ax.plot(x_grid, y_vals, label=f"{int(mat)}Y", color=col)

        ax.set_xlabel("(kappa - kappa*_USD) in bp")
        ax.set_ylabel("EUR–USD CSA value (USD)")
        ax.set_title(f"gamma = {g}")
        ax.legend(title="Maturity")
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 1 — Collateral Currency Valuation Adjustment", fontsize=13)
    plt.tight_layout()
    return pd.DataFrame(all_records), fig


def main():
    print("=" * 70)
    print("Experiment 2: Collateral Currency Valuation Effect")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    # Par margins for Table 3
    table3 = build_table3()
    print("\nTable 3 — Par margin change and CSA-switch value:")
    print(table3.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    table3.to_csv("results/exp2_table3.csv", index=False)

    # Figure 1
    fig1_data, fig = _build_figure1_both_gammas()
    fig.savefig("results/exp2_figure1.png", dpi=150)
    plt.close(fig)
    fig1_data.to_csv("results/exp2_figure1_data.csv", index=False)

    print("\nResults saved: results/exp2_table3.csv, results/exp2_figure1.png")
    return table3, fig1_data


def _build_figure1_both_gammas(
    maturities: list[float] = [1.0, 3.0, 5.0, 10.0],
    n_grid: int = 81,
    range_bp: float = 200.0,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Build Figure 1 for both gamma=0 and gamma=1 in side-by-side panels."""
    x_grid = np.linspace(-range_bp, range_bp, n_grid)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(maturities)))
    all_records = []

    for ax, g in zip(axes, [0, 1]):
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        for mat, col in zip(maturities, colors):
            dates, deltas = make_schedule(mat, freq=0.5)
            k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               g, 1.0, "USD", P)
            y_vals = []
            for x in x_grid:
                kappa = k_usd + x / 1e4
                dv = compute_collateral_value_adjustment(mat, g, kappa)
                y_vals.append(NOTIONAL * dv)
                all_records.append({
                    "maturity": mat, "gamma": g, "x_bp": x,
                    "kappa": kappa, "delta_V_USD": NOTIONAL * dv,
                })
            ax.plot(x_grid, y_vals, label=f"{int(mat)}Y", color=col)
        ax.set_xlabel("(kappa − kappa*_USD) [bp]")
        ax.set_ylabel("EUR − USD value [USD]")
        ax.set_title(f"gamma = {g}")
        ax.legend(title="Maturity")
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 1 — Collateral Currency Valuation Adjustment", fontsize=13)
    plt.tight_layout()
    return pd.DataFrame(all_records), fig


if __name__ == "__main__":
    # patch main to use corrected helper
    table3, fig1_data = main()
