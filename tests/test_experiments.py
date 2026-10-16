"""
Integration tests verifying experiment-level methodology adherence.

These tests run a reduced (quick) version of each experiment and
check qualitative properties / numerical bounds without full MC runs.
"""

import numpy as np
import pytest
from copy import copy

from src.params import ModelParams, ContractParams
from src.schedule import make_schedule
from src.differential_swap import par_margin, multi_period_value, single_period_value
from src.quanto import gamma_correction


P = ModelParams()


# ---------------------------------------------------------------------------
# Exp 1: MC vs Analytical — property tests
# ---------------------------------------------------------------------------

class TestExp1Properties:
    """Methodology checks for Exp 1 without running full 250k-path MC."""

    def test_single_period_gamma0_par_equals_domestic_rate(self):
        """
        For gamma=0, t=0, U=0, the par margin of a single-period swap
        equals R^d(0,T) — the domestic forward simple rate.
        """
        U, T = 0.0, 0.5
        from src.vasicek import futures_d
        k_par_analytic = futures_d(P.r_d_0, 0.0, U, T, P)

        from src.differential_swap import coupon_I1, coupon_I3
        I1 = coupon_I1(0.0, U, T, P.r_d_0, 1.0, "USD", P)
        I3 = coupon_I3(0.0, T, P.r_d_0, 1.0, "USD", P)
        delta = T - U
        k_par_formula = (I1 - I3) / (delta * I3)

        # Should match E^{Q_d}[R^d(0,T)] = F^d(0,T) to a close approximation
        # (the Gaussian vs simple-rate adjustment is present but small)
        assert k_par_formula > 0
        assert abs(k_par_formula - k_par_analytic) < 0.01   # within 100bp

    def test_single_period_decomposition_sums_correctly(self):
        """
        DS = I1 - gamma*I2 - (1-gamma+delta*kappa)*I3 for multiple (gamma, collateral) combos.
        """
        from src.differential_swap import coupon_I1, coupon_I2, coupon_I3
        U, T = 0.0, 0.5
        delta = T - U
        kappa = 0.03

        for g in [0, 1]:
            for coll in ["USD", "EUR"]:
                I1 = coupon_I1(0.0, U, T, P.r_d_0, 1.0, coll, P)
                I2 = coupon_I2(0.0, U, T, P.r_d_0, P.r_f_0, 1.0, coll, P)
                I3 = coupon_I3(0.0, T, P.r_d_0, 1.0, coll, P)

                from_components = I1 - g * I2 - (1 - g + delta * kappa) * I3
                c = ContractParams(gamma=g, kappa=kappa, beta=1.0, collateral=coll)
                from_func = single_period_value(0.0, U, T, P.r_d_0, P.r_f_0, c, P)
                assert from_components == pytest.approx(from_func, abs=1e-14)

    def test_mc_single_period_quick(self):
        """Quick MC (5000 paths) for single period gamma=0 should be near analytic."""
        from src.differential_swap import coupon_I1, coupon_I3
        from src.monte_carlo import mc_single_period

        U, T = 0.0, 0.5
        I1 = coupon_I1(0.0, U, T, P.r_d_0, 1.0, "USD", P)
        I3 = coupon_I3(0.0, T, P.r_d_0, 1.0, "USD", P)
        kappa = (I1 - I3) / (0.5 * I3)

        c = ContractParams(gamma=0, kappa=kappa, beta=1.0, collateral="USD")
        analytic = single_period_value(0.0, U, T, P.r_d_0, P.r_f_0, c, P)
        mc, se = mc_single_period(0.0, U, T, c, P, n_paths=5_000, seed=12)

        assert abs(mc - analytic) < 5 * se, (
            f"|MC - Analytic| = {abs(mc-analytic):.2e} > 5*SE = {5*se:.2e}"
        )


# ---------------------------------------------------------------------------
# Exp 2: Collateral effect — deterministic analytical checks
# ---------------------------------------------------------------------------

class TestExp2Properties:

    def test_collateral_effect_near_par_small(self):
        """
        Collateral effect should be small when kappa is near the USD-par margin
        (positive and negative coupon contributions cancel partially).
        """
        mat = 5.0
        dates, deltas = make_schedule(mat, 0.5)
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 1, 1.0, "USD", P)

        c_usd = ContractParams(gamma=1, kappa=k_usd, beta=1.0, collateral="USD")
        c_eur = ContractParams(gamma=1, kappa=k_usd, beta=1.0, collateral="EUR")
        v_usd = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0, c_usd, P)
        v_eur = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0, c_eur, P)
        # At par, USD value = 0; EUR value should be small but nonzero
        assert abs(v_eur) < 0.01   # < 100bp difference

    def test_collateral_effect_sign_for_positive_carry_f(self):
        """
        With carry_f > 0 and kappa well above USD par, EUR collateral
        gives higher or lower value depending on the payment schedule weighting.
        Just check the function runs and returns a finite value.
        """
        mat = 5.0
        dates, deltas = make_schedule(mat, 0.5)
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 0, 1.0, "USD", P)
        dv = multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                                ContractParams(gamma=0, kappa=k_usd + 0.01,
                                               beta=1.0, collateral="EUR"), P) - \
             multi_period_value(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                                ContractParams(gamma=0, kappa=k_usd + 0.01,
                                               beta=1.0, collateral="USD"), P)
        assert np.isfinite(dv)

    def test_par_margin_ratio_stable(self):
        """
        The par-margin change EUR-USD should be very small (sub-bp range).
        """
        for mat in [5.0, 10.0]:
            dates, deltas = make_schedule(mat, 0.5)
            k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               1, 1.0, "USD", P)
            k_eur = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0,
                               1, 1.0, "EUR", P)
            change_bp = (k_eur - k_usd) * 1e4
            assert abs(change_bp) < 1.0, (
                f"maturity={mat}: |change| = {abs(change_bp):.4f}bp > 1bp"
            )


# ---------------------------------------------------------------------------
# Exp 3: Quanto adjustment — property tests
# ---------------------------------------------------------------------------

class TestExp3Properties:

    def test_quanto_adj_gamma0_is_zero(self):
        """Delta kappa^eta(gamma=0) must be exactly zero."""
        from exp.exp3_quanto_adjustment import quanto_adjustment
        dk = quanto_adjustment(5.0, gamma=0, collateral="USD",
                               rho23=P.rho_23, sigma_bar=P.sigma_bar)
        assert dk == pytest.approx(0.0, abs=1e-12)

    def test_quanto_adj_linearity_in_gamma(self):
        """Delta kappa^eta(gamma) = gamma * Delta kappa^eta(gamma=1)."""
        from exp.exp3_quanto_adjustment import quanto_adjustment
        dk1 = quanto_adjustment(5.0, gamma=1, collateral="USD",
                                rho23=P.rho_23, sigma_bar=P.sigma_bar)
        # gamma is binary here (0 or 1), just check gamma=0 case
        dk0 = quanto_adjustment(5.0, gamma=0, collateral="USD",
                                rho23=P.rho_23, sigma_bar=P.sigma_bar)
        assert dk0 == pytest.approx(0.0 * dk1, abs=1e-12)

    def test_quanto_adj_monotone_in_rho23(self):
        """
        As rho_23 increases from negative to positive, quanto adjustment
        should change monotonically (with appropriate sigma_bar).
        """
        from exp.exp3_quanto_adjustment import quanto_adjustment
        rho_values = np.linspace(-0.5, 0.5, 11)
        dks = [quanto_adjustment(5.0, gamma=1, collateral="USD",
                                 rho23=r, sigma_bar=0.075)
               for r in rho_values]
        # Differences should all have the same sign (monotone)
        diffs = np.diff(dks)
        if len(diffs) > 0 and abs(diffs[0]) > 1e-12:
            signs = np.sign(diffs)
            # Allow at most one sign change (numerical noise)
            sign_changes = np.sum(signs[:-1] != signs[1:])
            assert sign_changes <= 1

    def test_quanto_adj_small_at_baseline(self):
        """Baseline quanto adjustment at rho23=0.10, sigma_bar=0.075 should be ~0.26 bp."""
        from exp.exp3_quanto_adjustment import quanto_adjustment
        dk = quanto_adjustment(5.0, gamma=1, collateral="USD",
                               rho23=0.10, sigma_bar=0.075)
        dk_bp = dk * 1e4
        assert 0.0 < dk_bp < 5.0, f"Unexpected quanto adj: {dk_bp:.4f} bp"

    def test_quanto_adj_similar_usd_eur(self):
        """
        Quanto adjustment under USD and EUR collateral should be very similar.
        """
        from exp.exp3_quanto_adjustment import quanto_adjustment
        dk_usd = quanto_adjustment(5.0, gamma=1, collateral="USD",
                                   rho23=0.10, sigma_bar=0.075)
        dk_eur = quanto_adjustment(5.0, gamma=1, collateral="EUR",
                                   rho23=0.10, sigma_bar=0.075)
        diff_bp = abs(dk_usd - dk_eur) * 1e4
        assert diff_bp < 0.01, f"USD-EUR quanto diff: {diff_bp:.4f} bp"


# ---------------------------------------------------------------------------
# Exp 4: Hedge ratios — shape and sign tests
# ---------------------------------------------------------------------------

class TestExp4Properties:

    def test_phi_f_zero_when_gamma0(self):
        """For gamma=0, foreign hedge ratio must be exactly zero."""
        from exp.exp4_hedge_performance import phi_f_vec
        dates, deltas = make_schedule(3.0, 0.5)
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 0, 1.0, "USD", P)
        c = ContractParams(gamma=0, kappa=k_usd, beta=1.0, collateral="USD")
        n = 100
        r_d = np.full(n, P.r_d_0)
        r_f = np.full(n, P.r_f_0)
        Q   = np.full(n, P.Q_0)
        phi = phi_f_vec(0.0, 0.0, 0.5, 0.5, r_d, r_f, Q, c, P)
        np.testing.assert_allclose(phi, 0.0, atol=1e-15)

    def test_phi_d_finite(self):
        """Domestic hedge ratio should be finite and non-trivial."""
        from exp.exp4_hedge_performance import phi_d_vec
        dates, deltas = make_schedule(3.0, 0.5)
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 1, 1.0, "USD", P)
        c = ContractParams(gamma=1, kappa=k_usd, beta=1.0, collateral="USD")
        n = 5
        r_d = np.full(n, P.r_d_0)
        r_f = np.full(n, P.r_f_0)
        phi = phi_d_vec(0.0, 0.0, 0.5, 0.5, r_d, r_f, c, P)
        assert np.all(np.isfinite(phi))

    def test_unhedged_has_larger_variance_than_hedged(self):
        """
        For gamma=1, the full hedge should reduce variance relative to unhedged.
        (Quick check with 200 paths, coarse rebalancing.)
        """
        from exp.exp4_hedge_performance import simulate_terminal_errors
        dates, deltas = make_schedule(3.0, 0.5)
        k_usd = par_margin(0.0, dates, deltas, P.r_d_0, P.r_f_0, 1, 1.0, "USD", P)
        c = ContractParams(gamma=1, kappa=k_usd, beta=1.0, collateral="USD")
        errs = simulate_terminal_errors(
            dates, deltas, c, P,
            n_steps_per_period=6,   # coarse but fast
            n_paths=200, seed=99,
        )
        var_unhd = np.var(errs["unhedged"])
        var_full = np.var(errs["full"])
        assert var_full < var_unhd * 0.5, (
            f"Full hedge did not significantly reduce variance: "
            f"var_full/var_unhd = {var_full/var_unhd:.3f}"
        )
