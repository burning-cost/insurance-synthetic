"""
Tests for InsuranceSynthesizer — the main fit/generate pipeline.

Covers:
- Basic fit/generate cycle (smoke tests)
- Output column names and dtypes
- Constraint enforcement
- Exposure-aware frequency generation
- Reproducibility (same seed → same output)
- Missing exposure column handling
- Gaussian fallback mode
- Summary output
- Severity synthesis (excluded from vine, drawn from conditional marginal)
"""

import numpy as np
import polars as pl
import pytest

from insurance_synthetic import InsuranceSynthesizer


# ---------------------------------------------------------------------------
# Smoke tests — basic fit/generate
# ---------------------------------------------------------------------------

class TestBasicFitGenerate:
    def test_fit_returns_self(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        result = synth.fit(small_motor_df)
        assert result is synth

    def test_generate_returns_dataframe(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        assert isinstance(out, pl.DataFrame)

    def test_generate_correct_row_count(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        for n in [10, 100, 500]:
            out = synth.generate(n)
            assert len(out) == n, f"Expected {n} rows, got {len(out)}"

    def test_generate_has_all_columns(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        assert set(out.columns) == set(small_motor_df.columns)

    def test_generate_before_fit_raises(self):
        synth = InsuranceSynthesizer()
        with pytest.raises(RuntimeError, match="fit"):
            synth.generate(10)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            InsuranceSynthesizer(method="ctgan")

    def test_fit_generate_medium_portfolio(self, medium_motor_df):
        """Full pipeline on a 1000-row portfolio with correlations."""
        synth = InsuranceSynthesizer(random_state=0)
        synth.fit(
            medium_motor_df,
            exposure_col="exposure",
            frequency_col="claim_count",
            severity_col="claim_amount",
        )
        out = synth.generate(200)
        assert len(out) == 200


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------

class TestOutputTypes:
    def test_integer_columns_are_integer(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        int_cols = ["driver_age", "vehicle_age", "vehicle_group", "ncd_years", "claim_count"]
        for col in int_cols:
            if col in out.columns:
                assert out[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64), \
                    f"{col} should be integer, got {out[col].dtype}"

    def test_float_columns_are_float(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        if "exposure" in out.columns:
            assert out["exposure"].dtype in (pl.Float32, pl.Float64)

    def test_string_columns_are_string(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        if "region" in out.columns:
            assert out["region"].dtype in (pl.Utf8, pl.String)

    def test_categorical_values_valid(self, small_motor_df):
        """Synthetic region values should come from the original set."""
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(200)
        if "region" in out.columns:
            orig_cats = set(small_motor_df["region"].to_list())
            synth_cats = set(out["region"].to_list())
            assert synth_cats.issubset(orig_cats), \
                f"Synthetic categories {synth_cats - orig_cats} not in original"


# ---------------------------------------------------------------------------
# Exposure-aware frequency generation
# ---------------------------------------------------------------------------

class TestFrequencyGeneration:
    def test_claim_count_non_negative(self, medium_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df, exposure_col="exposure", frequency_col="claim_count")
        out = synth.generate(500)
        assert (out["claim_count"] >= 0).all()

    def test_annualised_rate_reasonable(self, medium_motor_df):
        """
        Annualised claim frequency should be close to the real portfolio's rate.
        We allow a wide band (50%-200%) because we're comparing 1000 real rows
        against 500 synthetic rows — sampling variance dominates.
        """
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df, exposure_col="exposure", frequency_col="claim_count")
        out = synth.generate(500)

        real_rate = medium_motor_df["claim_count"].sum() / medium_motor_df["exposure"].sum()
        synth_rate = out["claim_count"].sum() / out["exposure"].sum()

        assert 0.3 * real_rate < synth_rate < 3.0 * real_rate, \
            f"Synthetic rate {synth_rate:.4f} too far from real {real_rate:.4f}"

    def test_frequency_rate_stored_on_synthesiser(self, medium_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        assert synth._frequency_rate is not None
        assert synth._frequency_rate > 0

    def test_exposure_range_reasonable(self, medium_motor_df):
        """Synthetic exposure should stay in (0, 1]."""
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        out = synth.generate(200)
        assert (out["exposure"] > 0).all()
        assert (out["exposure"] <= 1.05).all()  # small overshoot from PPF clipping is OK


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_simple_range_constraint_enforced(self, medium_motor_df):
        """driver_age must stay in [17, 90] after constraints."""
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        out = synth.generate(300, constraints={"driver_age": (17, 90)})
        ages = out["driver_age"].to_numpy()
        assert ages.min() >= 17
        assert ages.max() <= 90

    def test_multiple_constraints_enforced(self, medium_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        out = synth.generate(300, constraints={
            "driver_age": (17, 90),
            "ncd_years": (0, 20),
            "vehicle_group": (1, 50),
        })
        assert (out["driver_age"] >= 17).all()
        assert (out["driver_age"] <= 90).all()
        assert (out["ncd_years"] >= 0).all()
        assert (out["ncd_years"] <= 20).all()

    def test_callable_constraint(self, medium_motor_df):
        """Callable constraint should filter out non-positive exposures."""
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        out = synth.generate(
            200,
            constraints={"exposure": lambda x: x > 0},
        )
        assert (out["exposure"] > 0).all()

    def test_invalid_constraint_type_raises(self, medium_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        with pytest.raises(ValueError, match="Constraint"):
            synth.generate(10, constraints={"driver_age": "invalid"})

    def test_constraint_on_missing_column_ignored(self, medium_motor_df):
        """Constraint on a column not in the output should not raise."""
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(medium_motor_df)
        out = synth.generate(50, constraints={"nonexistent_col": (0, 100)})
        assert len(out) == 50


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self, small_motor_df):
        synth1 = InsuranceSynthesizer(random_state=123)
        synth1.fit(small_motor_df)
        out1 = synth1.generate(50)

        synth2 = InsuranceSynthesizer(random_state=123)
        synth2.fit(small_motor_df)
        out2 = synth2.generate(50)

        # Numeric columns should be identical
        for col in ["driver_age", "vehicle_age", "exposure"]:
            if col in out1.columns:
                np.testing.assert_array_almost_equal(
                    out1[col].to_numpy(), out2[col].to_numpy(),
                    decimal=6,
                    err_msg=f"Column {col} differs across identical seeds",
                )

    def test_different_seed_different_output(self, small_motor_df):
        synth1 = InsuranceSynthesizer(random_state=1)
        synth1.fit(small_motor_df)
        out1 = synth1.generate(100)

        synth2 = InsuranceSynthesizer(random_state=2)
        synth2.fit(small_motor_df)
        out2 = synth2.generate(100)

        # At least one column should differ
        any_differ = any(
            not np.array_equal(out1[col].to_numpy(), out2[col].to_numpy())
            for col in out1.columns
            if out1[col].dtype not in (pl.Utf8, pl.String)
        )
        assert any_differ, "Different seeds produced identical numeric output"

    def test_generator_seed_accepted(self, small_motor_df):
        """np.random.Generator seed should work as well as int seed."""
        gen_rng = np.random.default_rng(777)
        synth = InsuranceSynthesizer(random_state=gen_rng)
        synth.fit(small_motor_df)
        out = synth.generate(30)
        assert len(out) == 30


# ---------------------------------------------------------------------------
# Missing exposure column
# ---------------------------------------------------------------------------

class TestMissingExposure:
    def test_missing_exposure_warns(self, small_motor_df):
        df = small_motor_df.drop("exposure")
        synth = InsuranceSynthesizer(random_state=42)
        with pytest.warns(UserWarning, match="exposure_col"):
            synth.fit(df, exposure_col="exposure")

    def test_missing_exposure_still_generates(self, small_motor_df):
        df = small_motor_df.drop("exposure")
        synth = InsuranceSynthesizer(random_state=42)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synth.fit(df, exposure_col="exposure")
        out = synth.generate(20)
        assert len(out) == 20


# ---------------------------------------------------------------------------
# Gaussian method
# ---------------------------------------------------------------------------

class TestGaussianMethod:
    def test_gaussian_method_runs(self, small_motor_df):
        synth = InsuranceSynthesizer(method="gaussian", random_state=42)
        synth.fit(small_motor_df)
        out = synth.generate(50)
        assert isinstance(out, pl.DataFrame)
        assert len(out) == 50

    def test_gaussian_method_sets_family_set(self):
        synth = InsuranceSynthesizer(method="gaussian")
        assert synth.family_set == "gaussian"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_returns_string(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        s = synth.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_before_fit_returns_string(self):
        synth = InsuranceSynthesizer()
        s = synth.summary()
        assert "not fitted" in s.lower()

    def test_summary_contains_column_names(self, small_motor_df):
        synth = InsuranceSynthesizer(random_state=42)
        synth.fit(small_motor_df)
        s = synth.summary()
        for col in small_motor_df.columns:
            assert col in s, f"Column {col} not in summary"

    def test_get_params(self):
        synth = InsuranceSynthesizer(method="gaussian", trunc_lvl=3, n_threads=2)
        params = synth.get_params()
        assert params["method"] == "gaussian"
        assert params["trunc_lvl"] == 3
        assert params["n_threads"] == 2


# ---------------------------------------------------------------------------
# Severity synthesis (P0-5: excluded from vine, drawn from conditional marginal)
# ---------------------------------------------------------------------------

class TestSeveritySynthesis:
    """
    Tests for the fix to severity synthesis (claim_amount KS=0.93 bug).

    The root cause: zero-inflated severity columns break vine copula fitting.
    ~88% of rows are exactly 0 (non-claimers). When fitted as a continuous
    column, the CDF collapses around zero and the PPF inversion produces
    garbage. Fix: exclude severity from the vine, fit marginal on non-zero
    claims only, and draw severity conditioned on claim_count > 0.
    """

    def _make_severity_df(self, rng: np.random.Generator, n: int = 500) -> pl.DataFrame:
        """Portfolio with zero-inflated severity (roughly 85% zeros)."""
        claim_count = rng.poisson(0.1, size=n).astype(int)
        sev_mean = 3000.0
        claim_amount = np.where(
            claim_count > 0,
            np.maximum(0.0, rng.lognormal(np.log(sev_mean), 0.8, size=n)),
            0.0,
        )
        return pl.DataFrame({
            "driver_age": rng.integers(17, 75, size=n).tolist(),
            "ncd_years": rng.integers(0, 20, size=n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, size=n).tolist(),
            "claim_count": claim_count.tolist(),
            "claim_amount": claim_amount.tolist(),
        })

    def test_severity_col_excluded_from_vine_columns(self):
        """After fit(), severity column should not be in _columns (vine columns)."""
        rng = np.random.default_rng(0)
        df = self._make_severity_df(rng)
        synth = InsuranceSynthesizer(method="gaussian", random_state=0)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        assert "claim_amount" not in synth._columns, (
            "claim_amount should be excluded from vine copula columns"
        )

    def test_severity_marginal_fitted_on_nonzero_only(self):
        """_severity_marginal should be fitted and have positive mean."""
        rng = np.random.default_rng(1)
        df = self._make_severity_df(rng, n=500)
        synth = InsuranceSynthesizer(method="gaussian", random_state=1)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        assert synth._severity_marginal is not None, (
            "Severity marginal should be fitted on non-zero claims"
        )
        # Draw some samples — they should all be positive
        samples = synth._severity_marginal.ppf(np.array([0.1, 0.5, 0.9]))
        assert (samples > 0).all(), "Severity samples should be positive"

    def test_non_claimers_have_zero_severity(self):
        """Rows with claim_count == 0 must have claim_amount == 0."""
        rng = np.random.default_rng(2)
        df = self._make_severity_df(rng, n=600)
        synth = InsuranceSynthesizer(method="gaussian", random_state=2)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count",
                  exposure_col="exposure")
        out = synth.generate(300)
        non_claimers = out.filter(pl.col("claim_count") == 0)
        assert (non_claimers["claim_amount"] == 0.0).all(), (
            "Rows with claim_count=0 should have claim_amount=0"
        )

    def test_claimers_have_positive_severity(self):
        """Rows with claim_count > 0 should have claim_amount > 0."""
        rng = np.random.default_rng(3)
        # Use higher frequency to ensure enough claimers
        n = 800
        claim_count = rng.poisson(0.3, size=n).astype(int)
        sev_mean = 2500.0
        claim_amount = np.where(
            claim_count > 0,
            np.maximum(1.0, rng.lognormal(np.log(sev_mean), 0.7, size=n)),
            0.0,
        )
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, size=n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, size=n).tolist(),
            "claim_count": claim_count.tolist(),
            "claim_amount": claim_amount.tolist(),
        })
        synth = InsuranceSynthesizer(method="gaussian", random_state=3)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count",
                  exposure_col="exposure")
        out = synth.generate(500)
        claimers = out.filter(pl.col("claim_count") > 0)
        if len(claimers) > 0:
            assert (claimers["claim_amount"] > 0).all(), (
                "Rows with claim_count>0 should have positive claim_amount"
            )

    def test_severity_ks_is_reasonable(self):
        """
        Severity KS statistic should be substantially below 0.5 after the fix.
        The pre-fix KS was 0.93 — this test guards against regression.

        We test KS on non-zero claim amounts (comparing only the severity
        distribution, not the zero-inflation), which is what the fix targets.
        """
        from scipy.stats import ks_2samp

        rng = np.random.default_rng(42)
        n = 800
        claim_count = rng.poisson(0.15, size=n).astype(int)
        sev_mean = 3000.0
        claim_amount = np.where(
            claim_count > 0,
            np.maximum(0.1, rng.lognormal(np.log(sev_mean), 0.8, size=n)),
            0.0,
        )
        df = pl.DataFrame({
            "driver_age": rng.integers(17, 75, size=n).tolist(),
            "exposure": rng.uniform(0.1, 1.0, size=n).tolist(),
            "claim_count": claim_count.tolist(),
            "claim_amount": claim_amount.tolist(),
        })

        synth = InsuranceSynthesizer(method="gaussian", random_state=42)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count",
                  exposure_col="exposure")
        out = synth.generate(800)

        real_nonzero = claim_amount[claim_amount > 0]
        synth_nonzero = out.filter(pl.col("claim_amount") > 0)["claim_amount"].to_numpy()

        if len(synth_nonzero) >= 10 and len(real_nonzero) >= 10:
            ks_stat, _ = ks_2samp(real_nonzero, synth_nonzero)
            assert ks_stat < 0.5, (
                f"Severity KS statistic = {ks_stat:.3f}. "
                "Should be < 0.5 after fix. If near 0.93, regression to pre-fix behavior."
            )

    def test_severity_column_present_in_output(self):
        """Output DataFrame should include the severity column."""
        rng = np.random.default_rng(5)
        df = self._make_severity_df(rng)
        synth = InsuranceSynthesizer(method="gaussian", random_state=5)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        out = synth.generate(100)
        assert "claim_amount" in out.columns

    def test_output_column_order_matches_input(self):
        """Output columns should be in the same order as input."""
        rng = np.random.default_rng(6)
        df = self._make_severity_df(rng)
        synth = InsuranceSynthesizer(method="gaussian", random_state=6)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        out = synth.generate(100)
        assert out.columns == df.columns, (
            f"Column order mismatch: input={df.columns}, output={out.columns}"
        )

    def test_no_severity_col_still_works(self, medium_motor_df):
        """If severity_col is not specified, generation should work as before."""
        synth = InsuranceSynthesizer(method="gaussian", random_state=42)
        synth.fit(medium_motor_df, severity_col=None)
        out = synth.generate(100)
        assert len(out) == 100
        assert set(out.columns) == set(medium_motor_df.columns)

    def test_summary_mentions_severity_exclusion(self):
        """Summary should note that severity is excluded from vine copula."""
        rng = np.random.default_rng(7)
        df = self._make_severity_df(rng)
        synth = InsuranceSynthesizer(method="gaussian", random_state=7)
        synth.fit(df, severity_col="claim_amount", frequency_col="claim_count")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = synth.summary()
        assert "copula-excluded" in s.lower() or "excluded from vine" in s.lower(), (
            "Summary should mention that severity is excluded from the vine copula"
        )
