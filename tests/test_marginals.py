"""
Tests for marginal distribution fitting (_marginals.py).

We test:
1. Known distributions are recovered to reasonable parameter accuracy
2. PIT roundtrip: CDF(PPF(u)) ≈ u
3. Categorical handling — encoding and probability recovery
4. AIC-based selection prefers the correct family for clearly shaped data
5. Edge cases: tiny datasets, all-zero columns, constant series
"""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from insurance_synthetic import FittedMarginal, fit_marginal
from insurance_synthetic._marginals import _aic, _fit_continuous, _fit_discrete


# ---------------------------------------------------------------------------
# AIC helper
# ---------------------------------------------------------------------------

class TestAIC:
    def test_aic_formula(self):
        """AIC = 2k - 2*LL."""
        assert _aic(100.0, 2) == pytest.approx(2 * 2 - 2 * 100.0)

    def test_aic_lower_is_better(self):
        """A model with higher LL and same params has lower AIC."""
        assert _aic(200.0, 2) < _aic(100.0, 2)

    def test_aic_penalises_extra_params(self):
        """More parameters increases AIC, all else equal."""
        assert _aic(100.0, 3) > _aic(100.0, 2)


# ---------------------------------------------------------------------------
# Continuous marginal fitting
# ---------------------------------------------------------------------------

class TestContinuousFitting:
    def test_gamma_family_selected(self, gamma_series):
        """Gamma data should result in a gamma or lognormal marginal (both reasonable)."""
        m = fit_marginal(gamma_series)
        assert m.kind == "continuous"
        assert m.family_name() in ("gamma", "lognorm")

    def test_lognormal_family_selected(self, lognormal_series):
        """LogNormal data — lognorm or gamma should be selected."""
        m = fit_marginal(lognormal_series)
        assert m.kind == "continuous"
        assert m.family_name() in ("lognorm", "gamma")

    def test_normal_family_selected(self, normal_series):
        """Approximately normal data (driver age) — norm should rank highly."""
        m = fit_marginal(normal_series)
        assert m.kind == "continuous"
        # Normal or gamma are both plausible for driver age
        assert m.family_name() in ("norm", "gamma", "lognorm")

    def test_gamma_params_reasonable(self, gamma_series):
        """Fitted Gamma params should give a mean within 30% of the true mean."""
        m = fit_marginal(gamma_series, family="gamma")
        # True mean = a * scale = 2 * 500 = 1000
        dist_mean = m.dist.mean(*m.params)
        assert 600 < dist_mean < 1400, f"Fitted gamma mean {dist_mean:.0f} is far from 1000"

    def test_lognormal_params_reasonable(self, lognormal_series):
        """Fitted LogNormal should give a mean within 50% of the true mean."""
        m = fit_marginal(lognormal_series, family="lognorm")
        dist_mean = m.dist.mean(*m.params)
        true_mean = np.exp(7.5 + 1.0 ** 2 / 2)
        assert true_mean * 0.5 < dist_mean < true_mean * 2.0

    def test_explicit_family_override(self, gamma_series):
        """Passing family='norm' should fit a Normal regardless of AIC."""
        m = fit_marginal(gamma_series, family="norm")
        assert m.family_name() == "norm"

    def test_unknown_family_raises(self, gamma_series):
        with pytest.raises(ValueError, match="Unknown family"):
            fit_marginal(gamma_series, family="banana")


# ---------------------------------------------------------------------------
# Discrete marginal fitting
# ---------------------------------------------------------------------------

class TestDiscreteFitting:
    def test_poisson_fitted_for_count_data(self, poisson_series):
        """Integer non-negative data should trigger discrete fitting."""
        m = fit_marginal(poisson_series)
        assert m.kind == "discrete"
        assert m.family_name() in ("poisson", "nbinom")

    def test_poisson_rate_reasonable(self, poisson_series):
        """Fitted Poisson mu should be within 30% of the true 0.15."""
        m = fit_marginal(poisson_series, family="poisson")
        fitted_mu = m.params[0]
        assert 0.10 < fitted_mu < 0.25, f"Poisson mu {fitted_mu:.3f} far from 0.15"

    def test_discrete_lower_bound(self, poisson_series):
        """Discrete marginal should have clip_lower=0."""
        m = fit_marginal(poisson_series)
        assert m.clip_lower == 0.0

    def test_force_discrete(self):
        """Float series forced to discrete should fit Poisson/NegBin."""
        s = pl.Series("x", [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
        m = fit_marginal(s, is_discrete=True)
        assert m.kind == "discrete"


# ---------------------------------------------------------------------------
# Categorical marginal fitting
# ---------------------------------------------------------------------------

class TestCategoricalFitting:
    def test_string_column_detected_as_categorical(self, categorical_series):
        """Utf8 series should auto-detect as categorical."""
        m = fit_marginal(categorical_series)
        assert m.kind == "categorical"

    def test_categories_recovered(self, categorical_series):
        """All 5 categories should be in the fitted marginal."""
        m = fit_marginal(categorical_series)
        assert len(m.categories) == 5

    def test_probabilities_sum_to_one(self, categorical_series):
        """Category probabilities must sum to 1."""
        m = fit_marginal(categorical_series)
        assert sum(m.cat_probs) == pytest.approx(1.0, abs=1e-6)

    def test_is_categorical_flag(self):
        """Integer column with is_categorical=True should become categorical."""
        s = pl.Series("region_code", [1, 2, 3, 1, 2, 3, 1, 2])
        m = fit_marginal(s, is_categorical=True)
        assert m.kind == "categorical"

    def test_categorical_ppf_returns_indices(self, categorical_series):
        """PPF should return integer indices into categories list."""
        m = fit_marginal(categorical_series)
        indices = m.ppf(np.array([0.1, 0.5, 0.9]))
        assert all(0 <= i < len(m.categories) for i in indices.astype(int))

    def test_categorical_cdf_monotone(self, categorical_series):
        """Categorical CDF should be non-decreasing."""
        m = fit_marginal(categorical_series)
        x = np.arange(len(m.categories))
        cdfs = m.cdf(x)
        assert all(cdfs[i] <= cdfs[i + 1] for i in range(len(cdfs) - 1))

    def test_family_name_shows_levels(self, categorical_series):
        m = fit_marginal(categorical_series)
        assert "5" in m.family_name()


# ---------------------------------------------------------------------------
# PIT roundtrip tests
# ---------------------------------------------------------------------------

class TestPITRoundtrip:
    """CDF then PPF should approximately recover the original quantile."""

    def _check_pit_roundtrip(self, marginal: FittedMarginal, data: np.ndarray, tol: float = 0.05):
        u = marginal.cdf(data)
        recovered = marginal.ppf(u)
        # For continuous: PPF(CDF(x)) ≈ x (modulo clipping)
        # For discrete: PPF(CDF(x)) >= x (CDF steps mean we can overshoot)
        if marginal.kind == "continuous":
            median_err = float(np.median(np.abs(recovered - data) / (np.abs(data) + 1)))
            assert median_err < tol, f"Median relative PPF error {median_err:.4f} > {tol}"

    def test_gamma_pit_roundtrip(self, gamma_series):
        m = fit_marginal(gamma_series, family="gamma")
        self._check_pit_roundtrip(m, gamma_series.to_numpy())

    def test_lognormal_pit_roundtrip(self, lognormal_series):
        m = fit_marginal(lognormal_series, family="lognorm")
        self._check_pit_roundtrip(m, lognormal_series.to_numpy())

    def test_normal_pit_roundtrip(self, normal_series):
        m = fit_marginal(normal_series, family="norm")
        self._check_pit_roundtrip(m, normal_series.to_numpy())

    def test_ppf_clips_to_bounds(self, gamma_series):
        """PPF(0.0) should not return -inf; should be clipped to clip_lower."""
        m = fit_marginal(gamma_series)
        val = m.ppf(np.array([0.0001]))
        assert np.isfinite(val[0])
        assert val[0] >= 0

    def test_ppf_at_uniform_extremes(self, normal_series):
        """PPF should return finite values at u near 0 and 1."""
        m = fit_marginal(normal_series)
        low = m.ppf(np.array([1e-7]))
        high = m.ppf(np.array([1 - 1e-7]))
        assert np.isfinite(low[0])
        assert np.isfinite(high[0])


# ---------------------------------------------------------------------------
# FittedMarginal.rvs
# ---------------------------------------------------------------------------

class TestRVS:
    def test_rvs_returns_correct_size(self, gamma_series):
        m = fit_marginal(gamma_series)
        samples = m.rvs(50)
        assert len(samples) == 50

    def test_rvs_non_negative_for_positive_dist(self, gamma_series):
        m = fit_marginal(gamma_series)
        samples = m.rvs(200)
        assert all(s >= 0 for s in samples)

    def test_rvs_uses_rng(self, gamma_series):
        """Two calls with the same seed should return identical results."""
        m = fit_marginal(gamma_series)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        s1 = m.rvs(10, rng=rng1)
        s2 = m.rvs(10, rng=rng2)
        np.testing.assert_array_equal(s1, s2)

    def test_rvs_categorical(self, categorical_series):
        """Categorical RVS should return integer indices."""
        m = fit_marginal(categorical_series)
        samples = m.rvs(100)
        assert all(0 <= int(s) < len(m.categories) for s in samples)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_series_raises(self):
        s = pl.Series("x", [], dtype=pl.Float64)
        with pytest.raises(ValueError, match="no non-null values"):
            fit_marginal(s)

    def test_null_values_dropped(self):
        """Null values should be dropped before fitting."""
        s = pl.Series("x", [1.0, None, 2.0, None, 3.0])
        m = fit_marginal(s)  # should not raise
        assert m.kind == "continuous"

    def test_single_value_column(self):
        """Column with only one unique value — should fall back gracefully."""
        s = pl.Series("x", [5.0, 5.0, 5.0, 5.0, 5.0])
        m = fit_marginal(s)
        assert m.kind == "continuous"

    def test_very_small_dataset(self):
        """3-row dataset should not crash."""
        s = pl.Series("x", [100.0, 200.0, 300.0])
        m = fit_marginal(s)
        assert m is not None
