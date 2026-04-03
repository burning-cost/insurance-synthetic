"""Extended tests for marginal fitting — edge cases, error paths, PIT roundtrip."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from insurance_synthetic._marginals import (
    FittedMarginal,
    fit_marginal,
    _aic,
    _fit_continuous,
    _fit_discrete,
    _fit_negbin_mle,
)


# ---------------------------------------------------------------------------
# _aic
# ---------------------------------------------------------------------------

class TestAICExtended:
    def test_zero_params_allowed(self):
        # Degenerate but shouldn't crash
        v = _aic(10.0, 0)
        assert v == pytest.approx(-20.0)

    def test_negative_ll_gives_large_aic(self):
        # Large positive AIC is fine
        v = _aic(-1000.0, 2)
        assert v > 0

    def test_returns_float(self):
        v = _aic(50.0, 3)
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# _fit_negbin_mle
# ---------------------------------------------------------------------------

class TestFitNegBinMLE:
    def test_overdispersed_data(self):
        rng = np.random.default_rng(500)
        # NegBin with r=5, p=0.4 -> mean=7.5, var=18.75
        data = rng.negative_binomial(5, 0.4, size=500)
        n_hat, p_hat, ll = _fit_negbin_mle(data)
        assert n_hat > 0
        assert 0 < p_hat < 1
        assert np.isfinite(ll)

    def test_returns_three_values(self):
        rng = np.random.default_rng(501)
        data = rng.negative_binomial(3, 0.3, size=200)
        result = _fit_negbin_mle(data)
        assert len(result) == 3

    def test_handles_low_variance(self):
        # All-ones data (var ~ 0) should fall back gracefully
        data = np.ones(50, dtype=int)
        n_hat, p_hat, ll = _fit_negbin_mle(data)
        assert n_hat > 0
        assert 0 < p_hat < 1


# ---------------------------------------------------------------------------
# _fit_continuous
# ---------------------------------------------------------------------------

class TestFitContinuousExtended:
    def test_gamma_like_data(self):
        rng = np.random.default_rng(502)
        data = rng.gamma(3.0, 500.0, size=300)
        dist, params, aic = _fit_continuous(data)
        assert dist is not None
        assert np.isfinite(aic)

    def test_small_dataset_fallback(self):
        # < 4 points -> Normal fallback
        data = np.array([1.0, 2.0, 3.0])
        dist, params, aic = _fit_continuous(data)
        assert dist == stats.norm

    def test_all_same_values(self):
        data = np.ones(10) * 5.0
        dist, params, aic = _fit_continuous(data)
        # Should fall back to normal; just check it doesn't crash
        assert dist is not None


# ---------------------------------------------------------------------------
# _fit_discrete
# ---------------------------------------------------------------------------

class TestFitDiscreteExtended:
    def test_poisson_data(self):
        rng = np.random.default_rng(503)
        data = rng.poisson(0.15, size=500).astype(float)
        dist, params, aic = _fit_discrete(data)
        assert dist is not None
        assert np.isfinite(aic)

    def test_overdispersed_count_data(self):
        rng = np.random.default_rng(504)
        data = rng.negative_binomial(2, 0.3, size=500).astype(float)
        dist, params, aic = _fit_discrete(data)
        assert dist is not None


# ---------------------------------------------------------------------------
# FittedMarginal
# ---------------------------------------------------------------------------

class TestFittedMarginalExtended:
    def test_continuous_cdf_monotone(self):
        rng = np.random.default_rng(600)
        data = rng.gamma(2.0, 100.0, size=300)
        m = fit_marginal(pl.Series("x", data.tolist()))
        x_vals = np.percentile(data, [10, 25, 50, 75, 90])
        cdfs = m.cdf(x_vals)
        # CDF should be non-decreasing
        assert (np.diff(cdfs) >= -1e-8).all()

    def test_ppf_inverse_of_cdf(self):
        rng = np.random.default_rng(601)
        data = rng.lognormal(5.0, 0.5, size=200)
        m = fit_marginal(pl.Series("x", data.tolist()))
        u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ppf_vals = m.ppf(u)
        cdf_back = m.cdf(ppf_vals)
        np.testing.assert_allclose(cdf_back, u, atol=0.05)

    def test_rvs_shape(self):
        rng = np.random.default_rng(602)
        data = rng.normal(45, 10, size=200)
        m = fit_marginal(pl.Series("driver_age", data.tolist()))
        samples = m.rvs(100, rng=rng)
        assert len(samples) == 100

    def test_categorical_ppf_returns_valid_index(self):
        s = pl.Series("region", ["London", "North", "London", "South", "London"])
        m = fit_marginal(s)
        indices = m.ppf(np.array([0.1, 0.5, 0.9]))
        assert all(0 <= i < len(m.categories) for i in indices.astype(int))

    def test_categorical_cdf_at_last_category_near_one(self):
        s = pl.Series("region", ["A", "B", "C"] * 50)
        m = fit_marginal(s)
        # CDF at last index (2) should be 1.0
        assert m.cdf(np.array([len(m.categories) - 1]))[0] == pytest.approx(1.0, abs=1e-8)

    def test_family_name_continuous(self):
        rng = np.random.default_rng(603)
        data = rng.gamma(2.0, 100.0, size=200)
        m = fit_marginal(pl.Series("x", data.tolist()))
        name = m.family_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_family_name_categorical(self):
        s = pl.Series("cat", ["A", "B", "A"] * 30)
        m = fit_marginal(s)
        assert "Categorical" in m.family_name()

    def test_family_name_no_dist_returns_unknown(self):
        m = FittedMarginal(col_name="x", kind="continuous", dist=None)
        assert m.family_name() == "Unknown"

    def test_rvs_reproducible_with_seed(self):
        rng = np.random.default_rng(604)
        data = rng.normal(40, 12, size=300)
        m = fit_marginal(pl.Series("age", data.tolist()))
        s1 = m.rvs(50, rng=np.random.default_rng(0))
        s2 = m.rvs(50, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(s1, s2)

    def test_rvs_different_seeds_differ(self):
        rng = np.random.default_rng(605)
        data = rng.normal(40, 12, size=300)
        m = fit_marginal(pl.Series("age", data.tolist()))
        s1 = m.rvs(50, rng=np.random.default_rng(1))
        s2 = m.rvs(50, rng=np.random.default_rng(2))
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# fit_marginal — public API
# ---------------------------------------------------------------------------

class TestFitMarginalPublicAPI:
    def test_auto_continuous_returns_continuous_marginal(self):
        rng = np.random.default_rng(700)
        data = rng.gamma(2.0, 500.0, size=200)
        m = fit_marginal(pl.Series("claim_severity", data.tolist()))
        assert m.kind == "continuous"

    def test_auto_string_is_categorical(self):
        s = pl.Series("region", ["London", "North", "South"] * 30)
        m = fit_marginal(s)
        assert m.kind == "categorical"

    def test_auto_int_is_discrete(self):
        rng = np.random.default_rng(701)
        data = rng.poisson(0.2, size=300)
        m = fit_marginal(pl.Series("claim_count", data.tolist()))
        assert m.kind == "discrete"

    def test_explicit_family_gamma(self):
        rng = np.random.default_rng(702)
        data = rng.gamma(2.0, 100.0, size=200)
        m = fit_marginal(pl.Series("sev", data.tolist()), family="gamma")
        assert m.kind == "continuous"
        assert "gamma" in m.family_name()

    def test_explicit_family_lognorm(self):
        rng = np.random.default_rng(703)
        data = rng.lognormal(5.0, 1.0, size=200)
        m = fit_marginal(pl.Series("sev", data.tolist()), family="lognorm")
        assert "lognorm" in m.family_name()

    def test_explicit_family_norm(self):
        rng = np.random.default_rng(704)
        data = rng.normal(45, 12, size=300)
        m = fit_marginal(pl.Series("age", data.tolist()), family="norm")
        assert "norm" in m.family_name()

    def test_explicit_family_poisson(self):
        rng = np.random.default_rng(705)
        data = rng.poisson(0.15, size=500)
        m = fit_marginal(pl.Series("claims", data.tolist()), family="poisson")
        assert m.kind == "discrete"

    def test_invalid_family_raises(self):
        rng = np.random.default_rng(706)
        data = rng.normal(0, 1, size=100)
        with pytest.raises(ValueError, match="Unknown family"):
            fit_marginal(pl.Series("x", data.tolist()), family="xyz_not_real")

    def test_is_discrete_override(self):
        rng = np.random.default_rng(707)
        # Float series but forced discrete
        data = rng.exponential(1.0, size=200)
        m = fit_marginal(pl.Series("x", data.tolist()), is_discrete=True)
        assert m.kind == "discrete"

    def test_is_categorical_override(self):
        # Integer series forced to categorical
        s = pl.Series("vehicle_group", list(range(1, 11)) * 30)
        m = fit_marginal(s, is_categorical=True)
        assert m.kind == "categorical"

    def test_null_handling_drops_nulls(self):
        data = [1.0, None, 2.0, None, 3.0, 4.0, 5.0]
        s = pl.Series("x", data)
        m = fit_marginal(s)
        assert m.kind in ("continuous", "discrete")

    def test_col_name_preserved(self):
        rng = np.random.default_rng(708)
        data = rng.gamma(2.0, 100.0, size=200)
        m = fit_marginal(pl.Series("my_col_name", data.tolist()))
        assert m.col_name == "my_col_name"

    def test_aic_finite(self):
        rng = np.random.default_rng(709)
        data = rng.normal(0, 1, size=200)
        m = fit_marginal(pl.Series("x", data.tolist()))
        assert np.isfinite(m.aic)

    def test_categorical_probs_sum_to_one(self):
        s = pl.Series("region", ["London", "North", "South", "East"] * 25)
        m = fit_marginal(s)
        assert abs(m.cat_probs.sum() - 1.0) < 1e-8

    def test_categorical_categories_unique(self):
        s = pl.Series("region", ["A", "B", "A", "C", "B"] * 10)
        m = fit_marginal(s)
        assert len(m.categories) == len(set(m.categories))

    def test_clip_lower_non_negative_for_positive_data(self):
        rng = np.random.default_rng(710)
        data = rng.exponential(1.0, size=300)
        m = fit_marginal(pl.Series("x", data.tolist()))
        # clip_lower should be >= 0 for non-negative data
        assert m.clip_lower >= 0

    def test_nbinom_family_explicit(self):
        rng = np.random.default_rng(711)
        data = rng.negative_binomial(3, 0.3, size=400)
        m = fit_marginal(pl.Series("x", data.tolist()), family="nbinom")
        assert m.kind == "discrete"

    def test_empty_after_null_drop_raises(self):
        s = pl.Series("x", [None, None, None], dtype=pl.Float64)
        with pytest.raises(ValueError, match="no non-null"):
            fit_marginal(s)
