# AGENTS.md — Repository Knowledge for AI Agents

## Project Purpose
Closed-form pricing and Monte Carlo validation of USD-settled differential (cross-currency)
swaps under the Vasicek–Garman–Kohlhagen model. Implements Propositions 4.1, 4.2,
Corollary 4.1, and Proposition 5.2 from a pricing paper.

## Module Map

| File | Responsibility |
|---|---|
| `src/params.py` | `ModelParams` dataclass with all Table-1 parameters |
| `src/vasicek.py` | Bond prices `B^d/f`, conditional moments, futures expectations `F^d/f_t`, step covariance matrices for exact simulation |
| `src/quanto.py` | Closed-form Γ correction (eq. 4.9), loading functions σ_S(t), σ̂_S(t), helper integrals `_int_n_hat`, `_int_n_n_hat`, `_int_n_n` |
| `src/schedule.py` | `make_schedule(maturity, freq)` → (dates array, deltas array) |
| `src/differential_swap.py` | `coupon_terms_0()` for Prop 4.1; `multi_period_value()`, `par_margin()` for Prop 4.2/Cor 4.1 |
| `src/monte_carlo.py` | `run_mc_single_period()`, `run_mc_multi_period()` — exact stepwise Cholesky Gaussian simulation |
| `src/hedging.py` | `phi_d_ratios()`, `phi_f_ratios()`, `run_hedge_simulation()` — Euler-Maruyama simulation + Prop 5.2 rebalancing |
| `exp/exp1_*.py` | Table-2 style z-score validation |
| `exp/exp2_*.py` | Table-3 collateral effect + Figure-1 value adjustment curves |
| `exp/exp3_*.py` | Quanto adjustment sensitivity (Figure 2) |
| `exp/exp4_*.py` | Hedge RMSE table and Figure 3 rebalancing frequency sweep |

## Critical Bug Fixed
**`src/quanto.py` — sign error in `_int_n_hat` and `_int_n_n_hat`**

Both helpers computed `exp(-b*lo) − exp(-b*hi)` instead of the correct
`exp(-b*hi) − exp(-b*lo)` for the exponential sub-integral `t2`.  This made
`Γ_0(U,T)` 3–6× too large, producing a systematic z-score bias of ≈ −2 in the
EUR-collateral multi-period case (Exp 1, Case 4).

Fixed lines (search for `# CORRECTED`):
```python
# _int_n_hat: integral of n̂_V(u) over [lo, hi]
t2 = (exp(-b_hat*(V-hi)) - exp(-b_hat*(V-lo))) / b_hat   # correct sign

# _int_n_n_hat: integral of n_S(u)*n̂_V(u) over [lo, hi]
t2 = (exp(-b*(S-hi)) - exp(-b*(S-lo))) / b               # correct sign (r^d term)
t3 = (exp(-b_hat*(V-hi)) - exp(-b_hat*(V-lo))) / b_hat   # correct sign (r^f term)
```

Always verify Gamma against `scipy.integrate.quad` for any parameter change.

## Key Design Decisions

### Collateral discounting
`A^{β,i}_{t,S} = exp(-α^{β,i} * (S-t))` under constant-spread normalisation.  
Net-carry difference: `α^{β,f} − α^{β,d} = 0.0020` (20 bp).  
For USD collateral `α^{β,d}` is the reference; EUR adds 20 bp.

### Foreign futures under domestic measure
`F^f_0(U,T)` computed as `E^{Q_d}[1 + δ R^f(U,T)] / δ − 1/δ` using the fact that
`1 + δ R^f(U,T) = exp(∫_U^T r^f du)` is log-normal in the Vasicek model.  Formula:
```
E[exp(n̂(U,T)*r^f_U + C^f(U,T)) | F_0]
= exp(C^f(U,T) + n̂(U,T)*mu_U + 0.5*n̂(U,T)^2 * sig2_U)
```

### Exact MC simulation (Exp 1)
Step-by-step: given `(r^d_{T_{j-1}}, r^f_{T_{j-1}})`, draw
`(r^d_{T_j}, r^f_{T_j}, I^d_{j-1,j}, I^f_{j-1,j})` from their joint 4-variate
Gaussian distribution.  Covariance matrix built analytically in `vasicek.step_cov_matrix`.

### Euler simulation (Exp 4)
`dt = 1/252` (one trading day).  State: `(r^d, r^f, Q)` — 3-dimensional Euler.
Q enters hedge ratios through `phi^f_t = −(γ δ / Q_t) × A × B^d × Γ_t`.
Foreign futures gains use right-end conversion: `Q_{t+1} × ΔF^f`.

### Gamma rho23 sensitivity (Exp 3)
When sweeping `ρ₂₃`, keep foreign-measure drift `â = ĉ + σ̂ σ̄ ρ₂₃` fixed and
update domestic-measure drift `ĉ = â − σ̂ σ̄ ρ₂₃`. Failure to do this changes
the underlying model, not just the correlation.

## Test Coverage (67 tests)
- `test_vasicek.py` (27): bond prices, conditional moments, covariance matrix PSD,
  futures expectations
- `test_quanto.py` (11): loading functions, Gamma properties, no-correlation limit
- `test_differential_swap.py` (15): coupon terms, par margin identities,
  equal-carry consistency
- `test_experiments.py` (14): exp1 MC quick-check, exp2 collateral sign,
  exp3 linearity, exp4 gamma=0 φ^f=0

## Run Commands
```bash
# All tests
python -m pytest tests/ -v

# Individual experiments
python -m exp.exp1_pricing_validation   # ~60 s (250 000 MC paths)
python -m exp.exp2_collateral_effect    # <1 s
python -m exp.exp3_quanto_adjustment    # <1 s
python -m exp.exp4_hedge_performance    # ~2 min (Euler MC, 10 000 paths)
```

## Dependencies
`numpy`, `scipy`, `pandas`, `matplotlib` — all pre-installed.
No historical market data is fetched; no `MASSIVE_API_KEY` required.
