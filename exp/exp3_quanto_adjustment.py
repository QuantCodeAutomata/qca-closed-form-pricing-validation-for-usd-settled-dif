"""
Experiment 3: Isolation of the Contractual Quanto Adjustment.

Quantifies the deterministic Gamma correction's contribution to the par margin
of a 5Y USD-settled differential swap. Separates the quanto effect from
collateral by comparing with the no-quanto counterfactual (Gamma = 1).

Reproduces Figure 2: quanto adjustment vs rho_23 for multiple FX volatilities.
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

from src.params import ModelParams
from src.schedule import make_schedule
from src.differential_swap import par_margin


P = ModelParams()   # Table 1 baseline


# ---------------------------------------------------------------------------
# Quanto adjustment: Delta kappa = kappa*(Gamma) - kappa*(Gamma=1)
# ---------------------------------------------------------------------------

def quanto_adjustment(
    mat: float, gamma: int, collateral: str,
    rho23: float, sigma_bar: float,
    model: ModelParams | None = None,
) -> float:
    """
    Delta kappa_i^eta(gamma) = kappa_0^* - kappa_0^{*,no-quanto}

    in decimal units. Multiply by 1e4 for basis points.

    The foreign-measure drift a_hat = 0.05799 is held fixed while varying rho23.
    => c_hat is updated: c_hat = a_hat - sigma_hat * sigma_bar * rho23.

    Parameters
    ----------
    mat        : float  Maturity (years).
    gamma      : int    0 or 1.
    collateral : str    'USD' or 'EUR'.
    rho23      : float  New correlation rho_23.
    sigma_bar  : float  FX volatility.
    model      : ModelParams | None  If None, use global P.
    """
    if model is None:
        model = P

    p = copy(model)
    p.rho_23 = rho23
    p.sigma_bar = sigma_bar
    # Hold foreign-measure drift a_hat = 0.05799 fixed
    a_hat_fixed = 0.05799
    p.c_hat = a_hat_fixed - p.sigma_hat * sigma_bar * rho23

    dates, deltas = make_schedule(mat, freq=0.5)

    # Actual par margin (with Gamma)
    k_actual = par_margin(0.0, dates, deltas, p.r_d_0, p.r_f_0,
                          gamma, 1.0, collateral, p)
    # No-quanto counterfactual (Gamma = 1)
    k_noquanto = par_margin(0.0, dates, deltas, p.r_d_0, p.r_f_0,
                            gamma, 1.0, collateral, p,
                            gamma_override=1.0)
    return k_actual - k_noquanto


def baseline_quanto_check() -> dict:
    """
    Verify baseline: 5Y, gamma=1, rho23=0.10, sigma_bar=0.075.
    Expected: ~0.2570 bp USD, ~0.2569 bp EUR.
    """
    mat = 5.0
    results = {}
    for coll in ["USD", "EUR"]:
        dkappa = quanto_adjustment(mat, gamma=1, collateral=coll,
                                   rho23=0.10, sigma_bar=0.075)
        results[f"delta_kappa_{coll}_bp"] = dkappa * 1e4
    return results


def verify_linearity(mat: float = 5.0, collateral: str = "USD") -> dict:
    """
    Verify Delta kappa^eta(gamma) = gamma * Delta kappa^eta(1).
    """
    dk1 = quanto_adjustment(mat, gamma=1, collateral=collateral,
                            rho23=P.rho_23, sigma_bar=P.sigma_bar)
    dk0 = quanto_adjustment(mat, gamma=0, collateral=collateral,
                            rho23=P.rho_23, sigma_bar=P.sigma_bar)
    return {
        "dk_gamma1": dk1 * 1e4,
        "dk_gamma0": dk0 * 1e4,
        "gamma0_near_zero": abs(dk0) < 1e-10,
    }


def build_figure2(
    mat: float = 5.0,
    rho23_grid: np.ndarray | None = None,
    sigma_bars: list[float] = [0.05, 0.075, 0.10],
    n_grid: int = 101,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Figure 2: Quanto adjustment vs rho_23 for multiple FX volatilities.

    Two panels: USD and EUR collateral.
    """
    if rho23_grid is None:
        rho23_grid = np.linspace(-0.5, 0.5, n_grid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ls_map = ["-", "--", ":"]
    colors = ["steelblue", "firebrick", "forestgreen"]

    all_records = []
    for ax, coll in zip(axes, ["USD", "EUR"]):
        for sb, ls, col in zip(sigma_bars, ls_map, colors):
            y_vals = []
            for rho in rho23_grid:
                dk = quanto_adjustment(mat, gamma=1, collateral=coll,
                                       rho23=rho, sigma_bar=sb)
                y_vals.append(dk * 1e4)
                all_records.append({
                    "collateral": coll, "sigma_bar": sb,
                    "rho23": rho, "dk_bp": dk * 1e4,
                })
            label = f"σ_bar = {sb:.0%}"
            ax.plot(rho23_grid, y_vals, linestyle=ls, color=col, label=label)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlabel("rho_23 (correlation Z^2, Z^3)")
        ax.set_ylabel("Quanto adjustment (bp)")
        ax.set_title(f"Collateral: {coll}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 2 — Contractual Quanto Adjustment vs rho_23 (5Y swap)", fontsize=13)
    plt.tight_layout()
    return pd.DataFrame(all_records), fig


def main():
    print("=" * 70)
    print("Experiment 3: Isolation of the Contractual Quanto Adjustment")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    # Baseline verification
    base = baseline_quanto_check()
    print("\nBaseline quanto adjustment (rho23=0.10, sigma_bar=7.5%):")
    for k, v in base.items():
        print(f"  {k}: {v:.6f} bp")

    # Linearity check
    lin = verify_linearity()
    print("\nLinearity check (gamma=0 should give ~0 bp):")
    for k, v in lin.items():
        print(f"  {k}: {v}")

    # Figure 2
    fig2_data, fig = build_figure2()
    fig.savefig("results/exp3_figure2.png", dpi=150)
    plt.close(fig)
    fig2_data.to_csv("results/exp3_figure2_data.csv", index=False)

    # Summary table at baseline
    summary_rows = []
    for coll in ["USD", "EUR"]:
        dk = quanto_adjustment(5.0, gamma=1, collateral=coll,
                               rho23=P.rho_23, sigma_bar=P.sigma_bar)
        no_dk = quanto_adjustment(5.0, gamma=0, collateral=coll,
                                  rho23=P.rho_23, sigma_bar=P.sigma_bar)
        summary_rows.append({
            "collateral": coll,
            "gamma": 1,
            "dk_bp": dk * 1e4,
            "no_quanto_adj_bp": no_dk * 1e4,
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary table:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/exp3_summary.csv", index=False)

    print("\nResults saved: results/exp3_figure2.png, results/exp3_figure2_data.csv")
    return summary_df, fig2_data


if __name__ == "__main__":
    main()
