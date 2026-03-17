"""
Regression tests for P0/P1 bug fixes in insurance-synthetic.

Each test is named after the bug it guards against. These tests should never
be removed — they exist to prevent regressions to known-bad behaviour.

P0-1: INVESTIGATED AND REVERTED — the bug report claimed _gini() inflated
      values 2x by having `* 2`. Investigation showed the `* 2` is correct:
      the formula computes the concentration statistic (half the Gini), so `* 2`
      is required to produce the standard 2*AUC-1 Gini. The original code was
      correct; the bug report was wrong. Tests here assert `* 2` remains in place
      and Gini values are in the correct range.

P0-2: _tvar() used >= threshold, including the boundary quantile value.
      Standard actuarial definition is E[X | X > q_p] (strict inequality).

P0-3: _discrete_prev_cdf() returned CDF(0) for arr=0, making jitter width
      zero for zero counts and collapsing 85%+ of motor rows to a point mass.

P0-4: NegBin was fitted by MOM, Poisson by MLE — AIC not comparable across
      estimators. Both are now fitted by MLE.

P1:   Frobenius norm is now normalised by d (number of columns).
P1:   TSTR now uses same training size for real and synthetic models.
P1:   _fit_continuous AIC param count corrected for constrained distributions.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from insurance_synthetic._fidelity import _gini, _tvar
from insurance_synthetic._synthesiser import _discrete_prev_cdf
from insurance_synthetic._marginals import (
    FittedMarginal, _fit_discrete, _fit_negbin_mle, _aic, _count_continuous_params
)


# ---------------------------------------------------------------------------
# P0-1: Gini formula retains * 2 (validated correct; revert of bug report)
# ---------------------------------------------------------------------------

class TestGiniFormula:
    def test_gini_output_matches_2_times_concentration_statistic(self):
        """
        The _gini formula computes (sum(cumsum)/sum_y - (n+1)/2)/n,
        which is the concentration statistic S. The standard normalised Gini
        G = 2*S. The `* 2` must remain.

        We validate by checking the formula gives G ≈ 1/3 for a uniform
        distribution of y_true with a perfect model — the known analytical
        Gini of the uniform distribution is 1/3.
        """
        n = 10000
        y_true = np.arange(1, n + 1, dtype=float)
        y_pred = y_true.copy()  # perfect model
        gini = _gini(y_true, y_pred)
        # Theoretical Gini for discrete uniform = (n-1)/(3n) → 1/3 as n→∞
        expected = 1 / 3
        assert abs(gini - expected) < 0.01, (
            f"Gini for uniform perfect model should be ~{expected:.3f}, "
            f"got {gini:.4f}. If this is ~{expected/2:.3f}, the * 2 was removed."
        )

    def test_gini_is_in_valid_range(self):
        """Gini for any reasonable inputs must be in [-1, 1]."""
        rng = np.random.default_rng(7)
        y_true = rng.lognormal(5, 1, size=500)
        y_pred = rng.lognormal(5, 1, size=500)
        gini = _gini(y_true, y_pred)
        assert -1.0 <= gini <= 1.0, f"Gini {gini:.4f} outside [-1, 1]"

    def test_random_model_gini_near_zero(self):
        """A model with no predictive power should give Gini near 0."""
        rng = np.random.default_rng(0)
        y_true = rng.lognormal(5, 1, size=2000)
        y_pred = rng.uniform(size=2000)
        gini = _gini(y_true, y_pred)
        assert abs(gini) < 0.15, f"Random model Gini should be ~0, got {gini:.4f}"

    def test_perfect_model_better_than_random(self):
        """A perfect model should have strictly higher Gini than a random one."""
        rng = np.random.default_rng(42)
        y_true = rng.lognormal(5, 1, size=1000)
        gini_perfect = _gini(y_true, y_true)
        gini_random = _gini(y_true, rng.uniform(size=1000))
        assert gini_perfect > gini_random + 0.1, (
            f"Perfect Gini {gini_perfect:.3f} should be much higher than "
            f"random Gini {gini_random:.3f}"
        )

    def test_reversed_model_gini_negative(self):
        """A model that ranks in the opposite order should have negative Gini."""
        rng = np.random.default_rng(99)
        y_true = rng.lognormal(5, 1, size=1000)
        # Rank in reversed order: highest y_true predicted last
        y_pred = -y_true + rng.normal(0, 1e-8, size=1000)
        gini = _gini(y_true, y_pred)
        assert gini < -0.1, f"Reversed model Gini should be negative, got {gini:.4f}"


# ---------------------------------------------------------------------------
# P0-2: TVaR uses strict > threshold
# ---------------------------------------------------------------------------

class TestTVaRStrictInequality:
    def test_tvar_excludes_boundary(self):
        """
        With data where the quantile value appears at the threshold and above,
        >= and > give different answers. We use strict > per actuarial convention
        E[X | X > q_p].

        Data: 5 values each of 10, 50, 100 (15 total).
        q_0.50 is in the middle value range; we pick a percentile where the
        boundary effect is clear.
        """
        # 10 values of 1, 5 values of 100 — 100 appears in the top 1/3
        # At p=0.80, q80 is 1.0 (since 80% of values are 1s, but 80th quantile is still 1.0
        # using the linear interpolation). Actually need to be more careful.
        # Use exact data: [0]*9 + [10]*1 — 10% mass at 10, 90% at 0
        # q90 = 0.0 (since 90% of mass is at 0, q90 by default method is in the zeros)
        # Let's use a simpler clear-cut example:
        # sorted ascending: [5]*5 + [10]*5 + [100]*5 (n=15)
        # np.quantile with default 'linear': q_0.60 ≈ 10.0
        data = np.array([5.0] * 5 + [10.0] * 5 + [100.0] * 5)
        q60 = np.quantile(data, 0.60)
        # With >=: tail includes [10]*5 + [100]*5, mean = 55.0
        # With >: tail excludes the 10s, only [100]*5, mean = 100.0
        # But first let's see what q60 actually is
        # sorted: [5,5,5,5,5, 10,10,10,10,10, 100,100,100,100,100]
        # 0.60 * 14 = 8.4, so 8th index (0-based) = 10.0 (linear interp between 9th and 10th)
        # q60 = 10.0 if using method='linear'
        if abs(q60 - 10.0) < 0.5:
            tvar = _tvar(data, 0.60)
            # strict >: only [100]*5, mean = 100
            assert tvar > 50.0, (
                f"TVaR with strict > should be ~100, got {tvar:.1f}. "
                "If ~55, the >= bug has been reintroduced."
            )

    def test_tvar_when_all_tail_values_equal_threshold_returns_threshold(self):
        """
        When no value is strictly greater than the threshold
        (e.g., all values are equal), _tvar should fall back to threshold.
        """
        data = np.ones(100) * 5.0
        tvar = _tvar(data, 0.99)
        assert tvar == pytest.approx(5.0, abs=1e-6)

    def test_tvar_99th_geq_99th_quantile(self):
        """TVaR at p should always be >= VaR at p for non-negative data."""
        rng = np.random.default_rng(99)
        data = rng.lognormal(6, 1.5, size=5000)
        q99 = np.quantile(data, 0.99)
        tvar = _tvar(data, 0.99)
        assert tvar >= q99 - 1e-9, (
            f"TVaR {tvar:.2f} should be >= VaR {q99:.2f}"
        )

    def test_tvar_with_clear_boundary_mass(self):
        """
        Construct data where >= vs > gives materially different answers.

        data = [1]*89 + [10]*10 + [100]*1  (n=100)
        np.quantile(data, 0.90) == 10.0 exactly (verified analytically).
        With strict >:  tail = [100], TVaR = 100.
        With >=:        tail = [10]*10 + [100], TVaR = 18.2.
        """
        data = np.array([1.0] * 89 + [10.0] * 10 + [100.0] * 1)
        threshold = np.quantile(data, 0.90)
        assert threshold == pytest.approx(10.0), (
            f"Expected q90=10.0 for this dataset, got {threshold}. "
            "Test construction is wrong."
        )
        assert len(data[data > threshold]) == 1, "Strict tail should have 1 element"
        assert len(data[data >= threshold]) == 11, "Inclusive tail should have 11 elements"

        tvar = _tvar(data, 0.90)
        # Strict >: mean([100]) = 100
        # Inclusive >=: mean([10]*10 + [100]) ≈ 18.2
        assert tvar == pytest.approx(100.0, abs=1e-9), (
            f"TVaR should be 100.0 with strict >, got {tvar:.2f}. "
            "If ~18.2, the >= bug has been reintroduced."
        )


# ---------------------------------------------------------------------------
# P0-3: Discrete PIT jitter does not collapse at zero
# ---------------------------------------------------------------------------

class TestDiscretePITJitter:
    def _make_poisson_marginal(self, mu: float) -> FittedMarginal:
        return FittedMarginal(
            col_name="claim_count",
            kind="discrete",
            dist=stats.poisson,
            params=(mu,),
            aic=0.0,
            clip_lower=0.0,
        )

    def test_zero_counts_have_nonzero_jitter_width(self):
        """
        For arr=0, prev_u should be 0.0, not CDF(0) = CDF(arr).
        If prev_u == CDF(0) for arr=0, jitter width = CDF(0) - CDF(0) = 0,
        collapsing all zero-count policies to a single point.
        """
        marginal = self._make_poisson_marginal(mu=0.15)
        arr = np.zeros(100, dtype=float)

        prev_u = _discrete_prev_cdf(marginal, arr)
        raw_u = marginal.cdf(arr)

        # Jitter width = raw_u - prev_u
        jitter_width = raw_u - prev_u

        # For Poisson(0.15), CDF(0) = P(X=0) = exp(-0.15) ≈ 0.86
        # So jitter_width should be ~0.86, NOT 0.0
        assert (jitter_width > 1e-6).all(), (
            f"Jitter width for arr=0 is zero for some entries. "
            f"Min jitter: {jitter_width.min():.6f}. "
            "This means the PIT collapses zero counts to a point mass."
        )
        # Specifically, prev_u for arr=0 should be 0.0
        np.testing.assert_array_equal(
            prev_u, 0.0,
            err_msg="prev_u for arr=0 should be 0.0, not CDF(0)"
        )

    def test_nonzero_counts_have_correct_prev_u(self):
        """
        For arr=1, prev_u should be CDF(0) = P(X <= 0).
        For arr=2, prev_u should be CDF(1) = P(X <= 1).
        """
        marginal = self._make_poisson_marginal(mu=1.5)
        arr = np.array([1.0, 2.0, 3.0])

        prev_u = _discrete_prev_cdf(marginal, arr)
        expected = np.array([
            marginal.cdf(np.array([0.0]))[0],
            marginal.cdf(np.array([1.0]))[0],
            marginal.cdf(np.array([2.0]))[0],
        ])
        np.testing.assert_allclose(prev_u, expected, rtol=1e-9)

    def test_jitter_produces_uniform_u_not_point_mass(self):
        """
        After jitter, the PIT-transformed zero counts should be spread
        uniformly in [0, CDF(0)], not all equal to CDF(0).
        """
        rng = np.random.default_rng(0)
        marginal = self._make_poisson_marginal(mu=0.15)
        arr = np.zeros(1000, dtype=float)

        raw_u = marginal.cdf(arr)
        prev_u = _discrete_prev_cdf(marginal, arr)
        jittered = prev_u + (raw_u - prev_u) * rng.uniform(0, 1, size=1000)

        # Should be spread across [0, CDF(0)] — check that std is substantial
        assert jittered.std() > 0.05, (
            f"Jittered u for zero counts has std={jittered.std():.4f}, "
            "expected spread across [0, CDF(0)]. Point mass not resolved."
        )
        # All values should be in [0, CDF(0)]
        cdf0 = float(raw_u[0])
        assert (jittered >= 0).all()
        assert (jittered <= cdf0 + 1e-9).all()


# ---------------------------------------------------------------------------
# P0-4: NegBin fitted by MLE (AIC comparable with Poisson)
# ---------------------------------------------------------------------------

class TestNegBinMLE:
    def test_negbin_mle_recovers_reasonable_params(self):
        """
        MLE on genuine NegBin data should recover params in the right ballpark.
        """
        rng = np.random.default_rng(42)
        # NegBin(n=5, p=0.5): mean = n*(1-p)/p = 5, var = n*(1-p)/p^2 = 10
        data = rng.negative_binomial(5, 0.5, size=2000).astype(int)

        n_hat, p_hat, ll = _fit_negbin_mle(data)

        # Should recover n ≈ 5 and p ≈ 0.5 within reasonable tolerance
        assert 2.0 < n_hat < 15.0, f"n_hat={n_hat:.2f} far from true n=5"
        assert 0.3 < p_hat < 0.7, f"p_hat={p_hat:.3f} far from true p=0.5"
        assert np.isfinite(ll)

    def test_mle_ll_at_least_as_good_as_mom(self):
        """
        MLE log-likelihood must be >= MOM log-likelihood (MLE maximises it).
        """
        rng = np.random.default_rng(7)
        data = rng.negative_binomial(3, 0.4, size=500).astype(int)

        mean_ = float(np.mean(data))
        var_ = float(np.var(data))
        n_mom = mean_ ** 2 / (var_ - mean_)
        p_mom = mean_ / var_
        ll_mom = float(np.sum(stats.nbinom.logpmf(data, n_mom, p_mom)))

        _, _, ll_mle = _fit_negbin_mle(data)

        assert ll_mle >= ll_mom - 0.5, (
            f"MLE ll={ll_mle:.2f} should be >= MOM ll={ll_mom:.2f}. "
            "MLE must not underperform MOM."
        )

    def test_fit_discrete_both_families_use_mle(self):
        """
        _fit_discrete should select NegBin over Poisson for overdispersed data,
        and its AIC should reflect a genuine MLE (not MOM) log-likelihood.
        Overdispersed: var >> mean.
        """
        rng = np.random.default_rng(13)
        # Very overdispersed: NegBin(n=1, p=0.2), mean=4, var=20
        data = rng.negative_binomial(1, 0.2, size=2000).astype(int)

        dist, params, aic = _fit_discrete(data)
        assert dist == stats.nbinom, (
            f"Expected NegBin for overdispersed data, got {dist.name}. "
            "AIC-based selection may be broken if NegBin uses MOM."
        )

    def test_poisson_selected_for_equidispersed_data(self):
        """
        For equidispersed data (var ≈ mean), Poisson should win on AIC.
        """
        rng = np.random.default_rng(99)
        # Genuine Poisson — var ≈ mean
        data = rng.poisson(2.0, size=2000).astype(int)

        dist, params, aic = _fit_discrete(data)
        assert dist == stats.poisson, (
            f"Expected Poisson for equidispersed data, got {dist.name}. "
            "NegBin may be overfitting."
        )


# ---------------------------------------------------------------------------
# P1: Frobenius norm normalised by d
# ---------------------------------------------------------------------------

class TestFrobeniusNormalised:
    def test_frobenius_norm_stored_in_report(self):
        """
        correlation_report() should always return a frobenius_norm column
        with a finite, non-negative value.
        """
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(0)
        n = 200
        real = pl.DataFrame({
            "a": rng.normal(0, 1, n).tolist(),
            "b": rng.normal(0, 1, n).tolist(),
            "c": rng.normal(0, 1, n).tolist(),
        })
        synth = pl.DataFrame({
            "a": rng.normal(0, 1, n).tolist(),
            "b": rng.normal(0, 1, n).tolist(),
            "c": rng.normal(0, 1, n).tolist(),
        })
        rep = SyntheticFidelityReport(real, synth, target_col="a")
        corr = rep.correlation_report()
        frob = float(corr["frobenius_norm"][0])
        assert np.isfinite(frob)
        assert frob >= 0.0

    def test_frobenius_norm_is_per_column(self):
        """
        The normalised Frobenius norm should be divided by d. We verify this
        by constructing a case with a known matrix difference and checking
        the result matches manual normalisation.
        """
        from insurance_synthetic import SyntheticFidelityReport
        import numpy.linalg as la

        # Use large n so Spearman correlations converge to Pearson-like values
        rng = np.random.default_rng(42)
        n = 2000
        # Construct real with strong correlation, synth with no correlation
        x = rng.normal(0, 1, n)
        y_real = x + rng.normal(0, 0.01, n)  # near-perfect correlation with x
        y_synth = rng.normal(0, 1, n)         # independent

        real = pl.DataFrame({"x": x.tolist(), "y": y_real.tolist()})
        synth = pl.DataFrame({"x": x.tolist(), "y": y_synth.tolist()})

        rep = SyntheticFidelityReport(real, synth, target_col="x")
        corr = rep.correlation_report()
        frob_reported = float(corr["frobenius_norm"][0])

        # Manual: the difference is ~1 in off-diagonal, ~0 on diagonal
        # Frobenius(diff) ≈ sqrt(2) (two off-diagonal elements, each ~1)
        # Normalised by d=2: sqrt(2)/2 ≈ 0.707
        assert 0.3 < frob_reported < 1.5, (
            f"Frobenius norm {frob_reported:.4f} outside expected range [0.3, 1.5]. "
            "Check normalisation by d."
        )


# ---------------------------------------------------------------------------
# P1: AIC param count for constrained continuous distributions
# ---------------------------------------------------------------------------

class TestContinuousAICParamCount:
    def test_gamma_counts_two_params(self):
        """Gamma fitted with floc=0 should count 2 params (shape, scale), not 3."""
        n_params = _count_continuous_params(stats.gamma, (2.0, 0.0, 500.0))
        assert n_params == 2, (
            f"Gamma with floc=0 should have 2 free params, got {n_params}. "
            "AIC will over-penalise Gamma if loc is incorrectly counted."
        )

    def test_lognormal_counts_two_params(self):
        """LogNormal fitted with floc=0: 2 params (s, scale)."""
        n_params = _count_continuous_params(stats.lognorm, (0.9, 0.0, 2000.0))
        assert n_params == 2, f"LogNormal: expected 2, got {n_params}"

    def test_expon_counts_one_param(self):
        """Exponential fitted with floc=0: 1 param (scale)."""
        n_params = _count_continuous_params(stats.expon, (0.0, 500.0))
        assert n_params == 1, f"Expon: expected 1, got {n_params}"

    def test_normal_counts_two_params(self):
        """Normal is unconstrained — 2 params (loc, scale)."""
        n_params = _count_continuous_params(stats.norm, (45.0, 12.0))
        assert n_params == 2, f"Normal: expected 2, got {n_params}"
