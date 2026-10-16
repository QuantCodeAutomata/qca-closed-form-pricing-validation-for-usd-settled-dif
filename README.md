# USD-Settled Differential Swap Pricing — Vasicek–Garman–Kohlhagen Model

Closed-form pricing and hedging of USD-settled differential (cross-currency) swaps
under a two-factor Vasicek interest-rate model with log-normal FX dynamics.

## Research Objectives

Four experiments validate the paper's analytical results:

| # | Title | Key Output |
|---|---|---|
| 1 | Closed-form pricing validation | z-scores all within ±2 vs exact Gaussian MC |
| 2 | Collateral currency valuation effect | Table 3 par-margin changes & CSA-switch values |
| 3 | Contractual quanto adjustment | Figure 2 sensitivity to ρ₂₃ and σ̄ |
| 4 | Discrete rebalancing hedge performance | Table 4 RMSE; Figure 3 frequency scaling |

## Repository Structure

```
├── src/
│   ├── params.py            # Baseline model parameters (Table 1)
│   ├── vasicek.py           # Bond prices, conditional moments, futures expectations
│   ├── quanto.py            # Γ correction (eq. 4.9) and loading functions
│   ├── schedule.py          # Semiannual coupon schedule generation
│   ├── differential_swap.py # Propositions 4.1, 4.2, Corollary 4.1
│   ├── monte_carlo.py       # Exact Gaussian MC (step-wise Cholesky simulation)
│   └── hedging.py           # Proposition 5.2 futures hedge ratios & wealth engine
├── exp/
│   ├── exp1_pricing_validation.py
│   ├── exp2_collateral_effect.py
│   ├── exp3_quanto_adjustment.py
│   └── exp4_hedge_performance.py
├── tests/
│   ├── test_vasicek.py
│   ├── test_quanto.py
│   ├── test_differential_swap.py
│   └── test_experiments.py
├── results/
│   └── RESULTS.md           # Numerical findings
└── data/                    # (empty — no historical data required)
```

## Model

**Domestic short rate** (USD):
```
dr^d = (a − b r^d) dt + σ dZ¹
```

**Foreign short rate** (EUR, under domestic Q measure):
```
dr^f = (ĉ − b̂ r^f) dt + σ̂ dZ²
```

**FX dynamics**:
```
dQ = Q(r^d − r^f + λ^q) dt + Q σ̄ dZ³
```
with correlated Brownians `d⟨Zⁱ,Zʲ⟩ = ρᵢⱼ dt`.

**Baseline parameters** (paper Table 1):

| Parameter | Value |
|---|---|
| r^d_0 | 0.03670 |
| b, a | 3.0, 0.1101 |
| σ | 0.012 |
| r^f_0 | 0.01933 |
| b̂, ĉ | 3.0, 0.057915 |
| σ̂ | 0.010 |
| σ̄ (FX vol) | 0.075 |
| ρ₁₂, ρ₁₃, ρ₂₃ | 0.25, −0.25, 0.10 |
| Q₀ | 1.1782 |
| α^{c,f}−α^f | 0.0020 |

## Installation

```bash
pip install numpy scipy pandas matplotlib
```

All required packages are pre-installed in the standard container environment.

## Running Experiments

```bash
# Exp 1: Pricing validation (MC, ~60 s with 250 000 paths)
python -m exp.exp1_pricing_validation

# Exp 2: Collateral effect (analytical, <1 s)
python -m exp.exp2_collateral_effect

# Exp 3: Quanto adjustment sensitivity (analytical, <1 s)
python -m exp.exp3_quanto_adjustment

# Exp 4: Hedge performance (Euler MC, 10 000 paths, ~2 min)
python -m exp.exp4_hedge_performance
```

## Running Tests

```bash
python -m pytest tests/ -v
```

67 tests pass covering Vasicek moments, bond-price properties, Gamma-correction
correctness, analytical pricing identities, and hedge-ratio behaviour.

## Key Implementation Notes

- **Gamma correction** (`src/quanto.py`): integrals computed in closed form using
  Vasicek loading functions `σ_S(t) = −σ n(t,S)`, `σ̂_S(t) = −σ̂ n̂(t,S)`.
- **Exact Gaussian MC** (`src/monte_carlo.py`): step-by-step Cholesky simulation of
  `(r^d, r^f, ∫r^d, ∫r^f)` over coupon periods; no Euler discretisation in exp 1.
- **Hedge ratios** (`src/hedging.py`): Proposition 5.2 coupon-by-coupon ratios;
  foreign futures gains use right-end FX conversion `Q_{t+1} ΔF^f_{t+1}`.
- **Collateral discounting**: deterministic `A^{β,i}_{t,T} = exp(−α^{β,i}(T−t))`
  under constant spread normalisation, with net carry `α^{β,f}−α^{β,d} = 0.0020`.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full tables and figures.
