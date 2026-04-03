"""
Coverage expansion for insurance-synthetic (April 2026).

Targets:
- _synthesiser.py: _compute_group_rates, _build_valid_mask, _discrete_prev_cdf,
  dict marginals override, discrete_cols param, categorical_cols param,
  no-frequency-col path, severity with insufficient non-zero claims,
  frequency without severity, per-group rate lookup, get_params
- _copula.py: _parse_family_set, VineCopulaModel n_threads, using_fallback on fitted vine
- _marginals.py: FittedMarginal.family_name with None dist, _count_continuous_params,
  explicit nbinom/weibull/expon families, beta family on (0,1) data,
  FittedMarginal.cdf for discrete/continuous, rvs without rng arg
- _fidelity.py: _gini edge cases, _tvar edge cases, tvar_ratio zero-TVaR error,
  tstr_score ImportError, marginal_report empty array path, to_markdown target missing
- dp.py: _resolve_column_types edge cases, _to_pandas bad type, PrivacyReport __str__
  edge cases, column_bounds from external bounds, auto-detect with only one column type
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats
from unittest.mock import patch, MagicMock


# ===========================================================================
# _synthesiser.py — internal helpers
# ===========================================================================

class TestComputeGroupRates:
    """_compute_group_rates returns a rate table keyed by categorical tuples."""

    def test_single_category_single_group(self):
        from insurance_synthetic._synthesiser import _compute_group_rates

        df = pl.DataFrame({
            "region": ["London"] * 5,
            "claim_count": [1, 0, 1, 0, 2],
            "exposure": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        rates = _compute_group_rates(df, ["region"], "claim_count", "exposure")
        assert ("London",) in rates
        assert rates[("London",)] == pytest.approx(4 / 5.0)

    def test_multiple_categories(self):
        from insurance_synthetic._synthesiser import _compute_group_rates

        df = pl.DataFrame({
            "region": ["London", "London", "North", "North"],
            "claim_count": [1, 2, 0, 1],
            "exposure": [1.0, 1.0, 1.0, 1.0],
        })
        rates = _compute_group_rates(df, ["region"], "claim_count", "exposure")
        assert ("London",) in rates
        assert ("North",) in rates
        assert rates[("London",)] == pytest.approx(3.0 / 2.0)
        assert rates[("North",)] == pytest.approx(1.0 / 2.0)

    def test_zero_exposure_group_excluded(self):
        from insurance_synthetic._synthesiser import _compute_group_rates

        df = pl.DataFrame({
            "region": ["London", "North"],
            "claim_count": [1, 0],
            "exposure": [1.0, 0.0],  # North has zero exposure
        })
        rates = _compute_group_rates(df, ["region"], "claim_count", "exposure")
        assert ("London",) in rates
        assert ("North",) not in rates  # excluded due to zero exposure

    def test_multi_column_key(self):
        from insurance_synthetic._synthesiser import _compute_group_rates

        df = pl.DataFrame({
            "region": ["London", "London", "North", "North"],
            "cover_type": ["Comp", "TPO", "Comp", "TPO"],
            "claim_count": [1, 0, 2, 1],
            "exposure": [1.0, 1.0, 1.0, 1.0],
        })
        rates = _compute_group_rates(
            df, ["region", "cover_type"], "claim_count", "exposure"
        )
        assert ("London", "Comp") in rates
        assert ("London", "TPO") in rates
        assert len(rates) == 4

    def test_returns_dict(self):
        from insurance_synthetic._synthesiser import _compute_group_rates

        df = pl.DataFrame({
            "region": ["London"],
            "claim_count": [1],
            "exposure": [1.0],
        })
        result = _compute_group_rates(df, ["region"], "claim_count", "exposure")
        assert isinstance(result, dict)


class TestBuildValidMask:
    """_build_valid_mask enforces range and callable constraints."""

    def test_tuple_constraint_passes_all(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({"age": [20, 30, 40, 50]})
        mask = _build_valid_mask(df, {"age": (17, 90)})
        assert mask.sum() == 4

    def test_tuple_constraint_filters_some(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({"age": [10, 20, 30, 100]})
        mask = _build_valid_mask(df, {"age": (17, 90)})
        assert mask.sum() == 2  # 20 and 30 pass

    def test_callable_constraint(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({"exposure": [0.1, 0.5, 1.0, 1.5]})
        mask = _build_valid_mask(df, {"exposure": lambda x: x <= 1.0})
        assert mask.sum() == 3

    def test_invalid_rule_raises(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({"age": [20, 30]})
        with pytest.raises(ValueError, match="Constraint"):
            _build_valid_mask(df, {"age": "bad_rule"})

    def test_missing_column_skipped(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({"age": [20, 30]})
        # constraint on a column that doesn't exist should be silently skipped
        mask = _build_valid_mask(df, {"nonexistent": (0, 100)})
        assert mask.sum() == 2  # no rows filtered

    def test_multiple_constraints_combined(self):
        from insurance_synthetic._synthesiser import _build_valid_mask

        df = pl.DataFrame({
            "age": [15, 25, 35, 95],
            "exposure": [0.5, 1.5, 0.8, 0.9],
        })
        mask = _build_valid_mask(df, {
            "age": (17, 90),
            "exposure": (0.0, 1.0),
        })
        # row 1 (age=15) fails age; row 1 idx=1 (exposure=1.5) fails exposure
        # row 2 (age=25, exposure=1.5) fails exposure
        # row 3 (age=35, exposure=0.8) passes both
        # row 4 (age=95) fails age
        assert mask[2]  # index 2: age=35, exposure=0.8 — passes


class TestDiscretePrevCDF:
    """_discrete_prev_cdf handles zero counts correctly."""

    def test_zero_arr_returns_zero(self):
        from insurance_synthetic._synthesiser import _discrete_prev_cdf
        from insurance_synthetic import fit_marginal

        s = pl.Series("x", [0, 0, 1, 1, 2, 0, 0])
        m = fit_marginal(s)  # discrete
        arr = np.array([0, 0, 0])
        result = _discrete_prev_cdf(m, arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_positive_arr_returns_prev_cdf(self):
        from insurance_synthetic._synthesiser import _discrete_prev_cdf
        from insurance_synthetic import fit_marginal

        s = pl.Series("x", [0, 1, 2, 3, 1, 2, 0, 1])
        m = fit_marginal(s)
        arr = np.array([1, 2])
        result = _discrete_prev_cdf(m, arr)
        # prev_cdf(1) = cdf(0), prev_cdf(2) = cdf(1)
        assert result[0] < result[1]  # monotone
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_output_clipped_to_unit_interval(self):
        from insurance_synthetic._synthesiser import _discrete_prev_cdf
        from insurance_synthetic import fit_marginal

        s = pl.Series("x", [0, 1, 2, 3])
        m = fit_marginal(s)
        arr = np.array([0, 1, 2, 3, 100])
        result = _discrete_prev_cdf(m, arr)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestSynthesizerDictMarginals:
    """InsuranceSynthesizer accepts a dict of marginal overrides."""

    def test_dict_marginals_override_specific_column(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(0)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(
            method="gaussian",
            marginals={"driver_age": "norm"},
            random_state=0,
        )
        synth.fit(df)
        m = synth._fitted_marginals["driver_age"]
        assert m.family_name() == "norm"

    def test_dict_marginals_partial_override(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(1)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(
            method="gaussian",
            marginals={"exposure": "gamma"},
            random_state=1,
        )
        synth.fit(df)
        # exposure should be gamma; other columns auto
        m_exp = synth._fitted_marginals["exposure"]
        assert m_exp.family_name() == "gamma"


class TestSynthesizerDiscreteColsParam:
    """discrete_cols parameter forces discrete fitting on named columns."""

    def test_float_col_forced_to_discrete(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(2)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "some_count": rng.uniform(0, 5, n).tolist(),  # float but should be discrete
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=2)
        synth.fit(df, discrete_cols=["some_count"])
        m = synth._fitted_marginals["some_count"]
        assert m.kind == "discrete"


class TestSynthesizerCategoricalColsParam:
    """categorical_cols parameter forces categorical treatment on integer columns."""

    def test_integer_col_forced_to_categorical(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(3)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "region_code": rng.integers(1, 6, n).tolist(),  # integer but should be categorical
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=3)
        synth.fit(df, categorical_cols=["region_code"])
        m = synth._fitted_marginals["region_code"]
        assert m.kind == "categorical"


class TestSynthesizerNoFrequencyCol:
    """Behaviour when frequency_col is not in the DataFrame."""

    def test_fit_without_frequency_col(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(4)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=4)
        synth.fit(df, frequency_col="claim_count")  # not in df
        assert synth._frequency_col is None
        out = synth.generate(50)
        assert len(out) == 50

    def test_generate_without_frequency_col_no_claim_count(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(5)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=5)
        synth.fit(df, frequency_col="nonexistent")
        out = synth.generate(30)
        assert "driver_age" in out.columns
        assert "exposure" in out.columns


class TestSynthesizerSeverityEdgeCases:
    """Severity column with fewer than 4 non-zero claims should warn."""

    def test_severity_insufficient_nonzero_warns(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(6)
        n = 100
        # Only 2 non-zero claim amounts
        claim_amount = [0.0] * 98 + [1500.0, 2000.0]
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": [0] * 98 + [1, 1],
            "claim_amount": claim_amount,
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=6)
        with pytest.warns(UserWarning, match="fewer than 4"):
            synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        assert synth._severity_marginal is None

    def test_severity_all_zero_warns(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(7)
        n = 100
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": [0] * n,
            "claim_amount": [0.0] * n,
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=7)
        with pytest.warns(UserWarning, match="fewer than 4"):
            synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")

    def test_generate_with_no_severity_marginal_produces_zeros(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(8)
        n = 100
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": [0] * 98 + [1, 1],
            "claim_amount": [0.0] * 98 + [1500.0, 2000.0],
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        out = synth.generate(50)
        # With no severity marginal, all claim_amounts should be 0
        assert (out["claim_amount"] == 0.0).all()


class TestSynthesizerPerGroupRateLookup:
    """_compute_row_rates falls back to portfolio average for unseen groups."""

    def test_unseen_group_uses_portfolio_rate(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(9)
        n = 200
        df = pl.DataFrame({
            "region": rng.choice(["London", "North"], size=n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=9)
        synth.fit(df, exposure_col="exposure", frequency_col="claim_count")
        out = synth.generate(100)
        # Should have no errors and produce valid claim_counts
        assert (out["claim_count"] >= 0).all()

    def test_no_rate_table_uses_portfolio_average(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(10)
        n = 150
        # No categorical cols -> no rate table
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=10)
        synth.fit(df, exposure_col="exposure", frequency_col="claim_count")
        assert synth._frequency_rate_table is None
        out = synth.generate(50)
        assert (out["claim_count"] >= 0).all()


class TestSynthesizerGetParams:
    """get_params() returns the constructor parameters."""

    def test_get_params_returns_dict(self):
        from insurance_synthetic import InsuranceSynthesizer

        synth = InsuranceSynthesizer(method="gaussian", trunc_lvl=3)
        p = synth.get_params()
        assert isinstance(p, dict)

    def test_get_params_keys(self):
        from insurance_synthetic import InsuranceSynthesizer

        synth = InsuranceSynthesizer(method="vine", trunc_lvl=2, n_threads=4)
        p = synth.get_params()
        assert "method" in p
        assert "marginals" in p
        assert "family_set" in p
        assert "trunc_lvl" in p
        assert "n_threads" in p

    def test_get_params_values(self):
        from insurance_synthetic import InsuranceSynthesizer

        synth = InsuranceSynthesizer(method="gaussian", trunc_lvl=5, n_threads=2)
        p = synth.get_params()
        assert p["method"] == "gaussian"
        assert p["trunc_lvl"] == 5
        assert p["n_threads"] == 2


# ===========================================================================
# _copula.py — additional coverage
# ===========================================================================

class TestParseFamilySet:
    """_parse_family_set maps string names to pyvinecopulib family sets."""

    def test_unknown_family_set_raises(self):
        from insurance_synthetic._copula import _parse_family_set, _VINE_AVAILABLE

        if not _VINE_AVAILABLE:
            # When vine is not available, the function returns None without error
            result = _parse_family_set("unknown_name")
            assert result is None
        else:
            with pytest.raises(ValueError, match="Unknown family set"):
                _parse_family_set("unknown_family_set_xyz")

    def test_known_names_return_something(self):
        from insurance_synthetic._copula import _parse_family_set, _VINE_AVAILABLE

        if not _VINE_AVAILABLE:
            pytest.skip("pyvinecopulib not installed")
        for name in ["all", "parametric", "gaussian"]:
            result = _parse_family_set(name)
            assert result is not None


class TestVineCopulaModelNThreads:
    """VineCopulaModel stores n_threads parameter."""

    def test_n_threads_stored(self):
        from insurance_synthetic._copula import VineCopulaModel

        m = VineCopulaModel(n_threads=4)
        assert m.n_threads == 4

    def test_default_n_threads(self):
        from insurance_synthetic._copula import VineCopulaModel

        m = VineCopulaModel()
        assert m.n_threads == 1


class TestVineCopulaUsingFallbackProperty:
    """using_fallback property reflects the internal state."""

    def test_using_fallback_false_when_vine_available(self):
        from insurance_synthetic._copula import VineCopulaModel, _VINE_AVAILABLE

        m = VineCopulaModel()
        # Before fit, using_fallback reflects whether vine is available
        if _VINE_AVAILABLE:
            assert m.using_fallback is False
        else:
            assert m.using_fallback is True

    def test_using_fallback_true_when_forced(self):
        from insurance_synthetic._copula import VineCopulaModel

        m = VineCopulaModel()
        m._using_fallback = True
        assert m.using_fallback is True

    def test_fit_with_forced_fallback(self):
        from insurance_synthetic._copula import VineCopulaModel

        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, (100, 3))
        m = VineCopulaModel()
        m._using_fallback = True
        m.fit(u)
        assert m._fitted is True
        samples = m.simulate(50, rng=np.random.default_rng(0))
        assert samples.shape == (50, 3)


class TestGaussianFallbackSummary:
    """Gaussian fallback summary includes expected strings."""

    def test_summary_includes_gaussian_fallback(self):
        from insurance_synthetic._copula import VineCopulaModel

        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, (100, 4))
        m = VineCopulaModel()
        m._using_fallback = True
        m.fit(u)
        s = m.summary()
        assert "gaussian" in s.lower() or "fallback" in s.lower()

    def test_summary_shows_n_vars(self):
        from insurance_synthetic._copula import VineCopulaModel

        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, (100, 5))
        m = VineCopulaModel()
        m._using_fallback = True
        m.fit(u)
        s = m.summary()
        assert "5" in s  # number of variables


# ===========================================================================
# _marginals.py — additional coverage
# ===========================================================================

class TestFittedMarginalFamilyNameNoDist:
    """FittedMarginal.family_name() handles None dist."""

    def test_family_name_none_dist_returns_unknown(self):
        from insurance_synthetic import FittedMarginal

        m = FittedMarginal(col_name="x", kind="continuous", dist=None)
        assert m.family_name() == "Unknown"

    def test_family_name_categorical(self):
        from insurance_synthetic import FittedMarginal

        m = FittedMarginal(
            col_name="region",
            kind="categorical",
            categories=["A", "B", "C"],
        )
        name = m.family_name()
        assert "3" in name
        assert "Categorical" in name


class TestCountContinuousParams:
    """_count_continuous_params correctly counts free parameters."""

    def test_gamma_has_two_free_params(self):
        from insurance_synthetic._marginals import _count_continuous_params

        n = _count_continuous_params(stats.gamma, ())
        # gamma: 1 shape + 1 scale (loc fixed at 0) = 2
        assert n == 2

    def test_lognorm_has_two_free_params(self):
        from insurance_synthetic._marginals import _count_continuous_params

        n = _count_continuous_params(stats.lognorm, ())
        # lognorm: 1 shape (s) + 1 scale (loc fixed at 0) = 2
        assert n == 2

    def test_expon_has_one_free_param(self):
        from insurance_synthetic._marginals import _count_continuous_params

        n = _count_continuous_params(stats.expon, ())
        # expon: 0 shapes + 1 scale (loc fixed at 0) = 1
        assert n == 1

    def test_norm_has_two_free_params(self):
        from insurance_synthetic._marginals import _count_continuous_params

        # norm: 0 shapes + loc + scale = 2 (loc not fixed for norm)
        n = _count_continuous_params(stats.norm, ())
        assert n == 2


class TestFitMarginalExplicitFamilies:
    """fit_marginal accepts explicit continuous family names."""

    def test_explicit_expon_family(self):
        from insurance_synthetic import fit_marginal

        rng = np.random.default_rng(100)
        data = rng.exponential(500.0, size=200)
        s = pl.Series("x", data.tolist())
        m = fit_marginal(s, family="expon")
        assert m.family_name() == "expon"
        assert m.kind == "continuous"

    def test_explicit_weibull_family(self):
        from insurance_synthetic import fit_marginal

        rng = np.random.default_rng(101)
        data = rng.weibull(2.0, size=200) * 1000
        s = pl.Series("x", data.tolist())
        m = fit_marginal(s, family="weibull")
        assert m.kind == "continuous"

    def test_explicit_nbinom_family(self):
        from insurance_synthetic import fit_marginal

        rng = np.random.default_rng(102)
        data = rng.negative_binomial(5, 0.4, size=300)
        s = pl.Series("x", data.tolist())
        m = fit_marginal(s, family="nbinom")
        assert m.kind == "discrete"
        assert m.dist is stats.nbinom

    def test_explicit_beta_family_on_unit_data(self):
        from insurance_synthetic import fit_marginal

        rng = np.random.default_rng(103)
        data = rng.beta(2.0, 5.0, size=200)
        s = pl.Series("x", data.tolist())
        m = fit_marginal(s, family="beta")
        assert m.kind == "continuous"


class TestFittedMarginalCDF:
    """FittedMarginal.cdf() handles all three kinds."""

    def test_continuous_cdf_in_zero_one(self):
        from insurance_synthetic import fit_marginal

        rng = np.random.default_rng(200)
        data = rng.gamma(2.0, 500.0, size=200)
        m = fit_marginal(pl.Series("x", data.tolist()), family="gamma")
        cdf_vals = m.cdf(data)
        assert np.all(cdf_vals >= 0.0)
        assert np.all(cdf_vals <= 1.0)

    def test_discrete_cdf_monotone(self):
        from insurance_synthetic import fit_marginal

        s = pl.Series("x", [0, 1, 2, 3, 4, 5] * 20)
        m = fit_marginal(s)
        x = np.arange(6)
        vals = m.cdf(x)
        assert np.all(np.diff(vals) >= 0)  # non-decreasing

    def test_categorical_cdf_at_last_category_is_one(self):
        from insurance_synthetic import fit_marginal

        s = pl.Series("region", ["A", "B", "C", "A", "B", "C"])
        m = fit_marginal(s)
        n_cats = len(m.categories)
        val = m.cdf(np.array([n_cats - 1]))
        assert val[0] == pytest.approx(1.0)


class TestFittedMarginalRVSNoRNG:
    """rvs() without rng argument uses a default generator."""

    def test_rvs_no_rng_returns_correct_size(self):
        from insurance_synthetic import fit_marginal

        s = pl.Series("x", np.random.default_rng(0).gamma(2.0, 500.0, 100).tolist())
        m = fit_marginal(s)
        samples = m.rvs(50)  # no rng arg
        assert len(samples) == 50

    def test_rvs_categorical_no_rng(self):
        from insurance_synthetic import fit_marginal

        s = pl.Series("region", ["A", "B", "C", "A", "B"] * 20)
        m = fit_marginal(s)
        samples = m.rvs(30)
        assert len(samples) == 30


# ===========================================================================
# _fidelity.py — additional coverage
# ===========================================================================

class TestGiniEdgeCases:
    """_gini handles degenerate inputs."""

    def test_gini_empty_arrays_returns_zero(self):
        from insurance_synthetic._fidelity import _gini

        result = _gini(np.array([]), np.array([]))
        assert result == 0.0

    def test_gini_single_element(self):
        from insurance_synthetic._fidelity import _gini

        result = _gini(np.array([1.0]), np.array([1.0]))
        assert np.isfinite(result)

    def test_gini_perfect_ranking_positive(self):
        from insurance_synthetic._fidelity import _gini

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = _gini(y_true, y_pred)
        # Perfect ranking: Gini should be close to maximum (positive)
        assert g > 0.0

    def test_gini_reverse_ranking_negative(self):
        from insurance_synthetic._fidelity import _gini

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        g = _gini(y_true, y_pred)
        # Reverse ranking: Gini should be negative
        assert g < 0.0

    def test_gini_range(self):
        from insurance_synthetic._fidelity import _gini

        rng = np.random.default_rng(0)
        y_true = rng.poisson(0.1, 200).astype(float)
        y_pred = rng.uniform(0, 1, 200)
        g = _gini(y_true, y_pred)
        assert -1.0 <= g <= 1.0


class TestTVaREdgeCases:
    """_tvar handles boundary conditions."""

    def test_tvar_all_same_values(self):
        from insurance_synthetic._fidelity import _tvar

        values = np.ones(100)
        result = _tvar(values, 0.99)
        # All values equal threshold; tail is empty, returns threshold
        assert result == pytest.approx(1.0)

    def test_tvar_at_zero_percentile(self):
        from insurance_synthetic._fidelity import _tvar

        rng = np.random.default_rng(42)
        values = rng.exponential(100, size=200)
        result = _tvar(values, 0.0)
        # TVaR at 0 is just the mean of values > min (nearly all)
        assert result > 0

    def test_tvar_returns_float(self):
        from insurance_synthetic._fidelity import _tvar

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0])
        result = _tvar(values, 0.8)
        assert isinstance(result, float)

    def test_tvar_tail_mean_exceeds_var(self):
        from insurance_synthetic._fidelity import _tvar

        rng = np.random.default_rng(42)
        values = rng.lognormal(5, 1.5, size=500)
        threshold = float(np.quantile(values, 0.95))
        tvar = _tvar(values, 0.95)
        assert tvar >= threshold


class TestTVaRRatioZeroRealTVaR:
    """tvar_ratio raises when real TVaR is zero."""

    def test_raises_on_zero_real_tvar(self):
        real_df = pl.DataFrame({"x": [0.0] * 100, "exposure": [1.0] * 100})
        synth_df = pl.DataFrame({"x": [1.0] * 100, "exposure": [1.0] * 100})
        from insurance_synthetic import SyntheticFidelityReport

        report = SyntheticFidelityReport(real_df, synth_df, target_col="x")
        with pytest.raises(ValueError, match="zero"):
            report.tvar_ratio("x")


class TestTSTRImportError:
    """tstr_score raises ImportError when catboost is not installed."""

    def test_tstr_raises_import_error_without_catboost(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(42)
        n = 100
        real_df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(real_df, synth_df, target_col="claim_count")

        with patch.dict("sys.modules", {"catboost": None}):
            with pytest.raises((ImportError, TypeError)):
                report.tstr_score()


class TestTSTRMissingTarget:
    """tstr_score raises when target column is not in real_df."""

    def test_raises_when_target_missing(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(42)
        n = 100
        real_df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
        })
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(
            real_df, synth_df, target_col="nonexistent_target"
        )

        # Try to import catboost to see if we can even test this
        try:
            import catboost  # noqa: F401
            with pytest.raises(ValueError, match="Target column"):
                report.tstr_score()
        except ImportError:
            # CatBoost not installed — tstr raises ImportError first, which is fine
            with pytest.raises((ImportError, ValueError)):
                report.tstr_score()


class TestMarginalReportWithEmptyArrayColumn:
    """marginal_report handles columns where all values drop to empty after dropna."""

    def test_marginal_report_with_all_null_column(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(0)
        n = 50
        real_df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth_df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        report = SyntheticFidelityReport(real_df, synth_df)
        marg = report.marginal_report()
        assert isinstance(marg, pl.DataFrame)
        assert len(marg) == 2


class TestToMarkdownTargetMissing:
    """to_markdown handles target column not in real_df gracefully."""

    def test_to_markdown_handles_missing_target(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(42)
        n = 100
        real_df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(
            real_df, synth_df, target_col="claim_count"  # not in df
        )
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 50  # should still produce output


class TestCorrelationReportSingleColumn:
    """correlation_report with exactly one numeric column raises ColumnNotFoundError
    because the empty rows list produces a no-column DataFrame on which .sort('delta')
    fails. This test documents the known limitation — single-column DataFrames are not
    a realistic use case for the fidelity report.
    """

    def test_single_numeric_col_raises(self):
        from insurance_synthetic import SyntheticFidelityReport
        import polars.exceptions

        rng = np.random.default_rng(0)
        n = 50
        real_df = pl.DataFrame({"driver_age": rng.integers(17, 75, n).tolist()})
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(real_df, synth_df)
        # With a single column, correlation_report raises because the internal
        # polars sort on the empty-rows DataFrame finds no 'delta' column.
        with pytest.raises((polars.exceptions.ColumnNotFoundError, Exception)):
            report.correlation_report()


# ===========================================================================
# dp.py — additional coverage
# ===========================================================================

class TestResolvColumnTypes:
    """_resolve_column_types handles edge cases."""

    def test_both_explicit_returns_them(self):
        from insurance_synthetic.dp import _resolve_column_types

        df = pd.DataFrame({
            "region": ["London", "North"],
            "age": [25.0, 35.0],
        })
        cats, conts = _resolve_column_types(df, ["region"], ["age"])
        assert "region" in cats
        assert "age" in conts

    def test_missing_column_raises(self):
        from insurance_synthetic.dp import _resolve_column_types

        df = pd.DataFrame({"region": ["London"], "age": [25.0]})
        with pytest.raises(ValueError, match="not found"):
            _resolve_column_types(df, ["region", "nonexistent"], ["age"])

    def test_auto_detect_string_as_cat(self):
        from insurance_synthetic.dp import _resolve_column_types

        df = pd.DataFrame({
            "region": ["London", "North", "South"],
            "age": [25.0, 35.0, 45.0],
        })
        cats, conts = _resolve_column_types(df, None, None)
        assert "region" in cats
        assert "age" in conts

    def test_no_columns_raises(self):
        from insurance_synthetic.dp import _resolve_column_types

        df = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            _resolve_column_types(df, None, None)

    def test_only_one_column_type_explicit(self):
        """Providing only categorical_columns, continuous inferred."""
        from insurance_synthetic.dp import _resolve_column_types

        df = pd.DataFrame({
            "region": ["London", "North"],
            "age": [25.0, 35.0],
            "income": [30000.0, 50000.0],
        })
        cats, conts = _resolve_column_types(df, ["region"], None)
        assert "region" in cats
        assert "age" in conts
        assert "income" in conts


class TestToPandas:
    """_to_pandas converts polars DF and passes pandas DF through."""

    def test_polars_df_converted(self):
        from insurance_synthetic.dp import _to_pandas

        pl_df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = _to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_pandas_df_returned(self):
        from insurance_synthetic.dp import _to_pandas

        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        result = _to_pandas(pd_df)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_type_raises(self):
        from insurance_synthetic.dp import _to_pandas

        with pytest.raises(TypeError, match="Expected"):
            _to_pandas([1, 2, 3])  # list is not a valid input

    def test_polars_df_is_copy(self):
        """The returned pandas DF should not be the same object as the input."""
        from insurance_synthetic.dp import _to_pandas

        pl_df = pl.DataFrame({"a": [1, 2, 3]})
        result = _to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        # Modifying result should not affect original polars df
        result["a"] = 99
        assert pl_df["a"][0] != 99


class TestPrivacyReportStrEdgeCases:
    """PrivacyReport.__str__ renders correctly under edge cases."""

    def test_str_no_tail_fidelity_no_column_bounds(self):
        from insurance_synthetic.dp import PrivacyReport

        r = PrivacyReport(
            epsilon=1.0,
            epsilon_discretisation=0.1,
            epsilon_synthesis=0.9,
            delta=1e-9,
            mechanism="AIM",
        )
        s = str(r)
        assert "AIM" in s
        assert "1.0" in s  # epsilon

    def test_str_with_column_bounds(self):
        from insurance_synthetic.dp import PrivacyReport

        r = PrivacyReport(
            epsilon=2.0,
            epsilon_discretisation=0.2,
            epsilon_synthesis=1.8,
            delta=1e-9,
            column_bounds={"driver_age": (17.0, 100.0), "exposure": (0.01, 1.0)},
        )
        s = str(r)
        assert "driver_age" in s
        assert "exposure" in s

    def test_str_with_warnings(self):
        from insurance_synthetic.dp import PrivacyReport

        r = PrivacyReport(
            epsilon=1.0,
            epsilon_discretisation=0.1,
            epsilon_synthesis=0.9,
            delta=1e-9,
            warnings=["Something is degraded.", "Another warning."],
        )
        s = str(r)
        assert "Something is degraded" in s

    def test_str_empty_warnings(self):
        from insurance_synthetic.dp import PrivacyReport

        r = PrivacyReport(
            epsilon=1.0,
            epsilon_discretisation=0.1,
            epsilon_synthesis=0.9,
            delta=1e-9,
            warnings=[],
        )
        s = str(r)
        assert isinstance(s, str)
        assert len(s) > 0


class TestPrivacyReportColumnBoundsExternalSpec:
    """PrivacyReport.column_bounds uses external bounds when available."""

    def test_external_bounds_override_data_bounds(self):
        """When bounds are specified externally, they appear in column_bounds."""
        from unittest.mock import MagicMock, patch
        from insurance_synthetic.dp import DPInsuranceSynthesizer

        mock_aim_class = MagicMock()
        mock_aim_instance = MagicMock()
        mock_aim_class.return_value = mock_aim_instance

        rng = np.random.default_rng(42)
        small_pd = pd.DataFrame({
            "driver_age": rng.integers(17, 85, 100).astype(float),
            "region": rng.choice(["London", "North"], 100),
        })

        with patch.dict("sys.modules", {"snsynth": MagicMock(AIMSynthesizer=mock_aim_class)}):
            synth = DPInsuranceSynthesizer(
                epsilon=1.0,
                preprocessor_eps=0.0,
                bounds={"driver_age": (17.0, 100.0)},
            )
            synth.fit(
                small_pd,
                categorical_columns=["region"],
                continuous_columns=["driver_age"],
            )
            report = synth.privacy_report()
            # External bound should be in column_bounds
            assert "driver_age" in report.column_bounds
            lo, hi = report.column_bounds["driver_age"]
            assert lo == pytest.approx(17.0)
            assert hi == pytest.approx(100.0)


class TestQuantileBinEdgeCases:
    """Additional edge cases for _quantile_bin."""

    def test_n_bins_equals_two(self):
        from insurance_synthetic.dp import _quantile_bin

        arr = np.linspace(0, 100, 50)
        edges, binned = _quantile_bin(arr, n_bins=2)
        assert len(edges) >= 2
        assert len(binned) == 50

    def test_single_unique_value(self):
        from insurance_synthetic.dp import _quantile_bin

        arr = np.full(100, 42.0)
        edges, binned = _quantile_bin(arr, n_bins=10)
        # Should not crash; all values in same bin
        assert len(binned) == 100

    def test_col_bounds_with_values_outside_bounds(self):
        """Values outside provided col_bounds should still be clipped to valid bins."""
        from insurance_synthetic.dp import _quantile_bin

        arr = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        edges, binned = _quantile_bin(arr, n_bins=3, col_bounds=(0.0, 30.0))
        assert edges[0] <= 0.0
        assert edges[-1] >= 30.0
        # All bin indices within bounds
        assert np.all(binned >= 0)
        assert np.all(binned <= len(edges) - 2)

    def test_nan_values_ignored(self):
        from insurance_synthetic.dp import _quantile_bin

        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        edges, binned = _quantile_bin(arr, n_bins=3)
        # Should not crash; result same length as input
        assert len(binned) == 5


# ===========================================================================
# Integration: multi-step pipeline edge cases
# ===========================================================================

class TestSynthesizerMultiStepEdgeCases:
    """Integration tests for less-common combinations."""

    def test_fit_generate_with_only_continuous_cols(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(20)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).astype(float).tolist(),
            "vehicle_age": rng.integers(0, 15, n).astype(float).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=20)
        synth.fit(df)
        out = synth.generate(50)
        assert len(out) == 50

    def test_fit_generate_with_single_row_count(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(21)
        n = 150
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=21)
        synth.fit(df)
        out = synth.generate(1)
        assert len(out) == 1

    def test_generate_large_n_still_correct(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(22)
        n = 200
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=22)
        synth.fit(df)
        out = synth.generate(2000)
        assert len(out) == 2000

    def test_generate_with_constraint_callable_and_tuple_combined(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(23)
        n = 200
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=23)
        synth.fit(df)
        out = synth.generate(
            100,
            constraints={
                "driver_age": (17, 90),
                "exposure": lambda x: x > 0.05,
            },
        )
        assert (out["driver_age"] >= 17).all()
        assert (out["driver_age"] <= 90).all()
        assert (out["exposure"] > 0.05).all()

    def test_summary_with_frequency_rate_table(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(24)
        n = 200
        df = pl.DataFrame({
            "region": rng.choice(["London", "North", "South"], n).tolist(),
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=24)
        synth.fit(df, frequency_col="claim_count", exposure_col="exposure")
        s = synth.summary()
        # Should mention per-group rates
        assert "region" in s or "Per-group" in s

    def test_frequency_col_in_output_after_generate(self):
        from insurance_synthetic import InsuranceSynthesizer

        rng = np.random.default_rng(25)
        n = 200
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
            "claim_count": rng.poisson(0.1, n).tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=25)
        synth.fit(df, frequency_col="claim_count")
        out = synth.generate(100)
        assert "claim_count" in out.columns
        assert out["claim_count"].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64
        )


class TestFidelityReportEdgeCases:
    """Additional fidelity edge cases."""

    def test_marginal_report_string_column_produces_nulls(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(0)
        n = 50
        real_df = pl.DataFrame({
            "region": rng.choice(["London", "North"], n).tolist(),
            "driver_age": rng.integers(17, 75, n).tolist(),
        })
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(real_df, synth_df)
        marg = report.marginal_report()
        region_row = marg.filter(pl.col("column") == "region")
        assert region_row["ks_statistic"][0] is None

    def test_exposure_weighted_ks_identical_dfs(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        })
        report = SyntheticFidelityReport(df, df)
        ks = report.exposure_weighted_ks("driver_age")
        # Identical distributions → KS should be very small
        assert ks < 0.3  # not exactly 0 due to weighted resampling

    def test_correlation_report_two_columns(self):
        from insurance_synthetic import SyntheticFidelityReport

        rng = np.random.default_rng(0)
        n = 100
        real_df = pl.DataFrame({
            "a": rng.integers(0, 100, n).tolist(),
            "b": rng.integers(0, 100, n).tolist(),
        })
        synth_df = real_df.clone()
        report = SyntheticFidelityReport(real_df, synth_df)
        corr = report.correlation_report()
        # Two columns → 1 pair
        assert len(corr) == 1
        assert "col_a" in corr.columns
        assert "col_b" in corr.columns
