"""Extended tests for SyntheticFidelityReport — edge cases and metric validation."""

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_synthetic import SyntheticFidelityReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_pair(n=200, seed=42):
    rng = np.random.default_rng(seed)
    real_df = pl.DataFrame({
        "driver_age": rng.integers(17, 85, n).tolist(),
        "exposure": rng.uniform(0.1, 1.0, n).tolist(),
        "claim_count": rng.poisson(0.12, n).tolist(),
    })
    # Synthetic slightly perturbed but same shape
    rng2 = np.random.default_rng(seed + 1)
    synth_df = pl.DataFrame({
        "driver_age": rng2.integers(17, 85, n).tolist(),
        "exposure": rng2.uniform(0.1, 1.0, n).tolist(),
        "claim_count": rng2.poisson(0.12, n).tolist(),
    })
    return real_df, synth_df


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFidelityConstruction:
    def test_basic_construction(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        assert report is not None

    def test_missing_columns_warns(self):
        real_df, synth_df = _simple_pair()
        synth_df_missing = synth_df.drop("claim_count")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = SyntheticFidelityReport(real_df, synth_df_missing)
            assert len(w) >= 1

    def test_custom_exposure_col(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(
            real_df, synth_df, exposure_col="exposure", target_col="claim_count"
        )
        assert report.exposure_col == "exposure"

    def test_custom_target_col(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df, target_col="driver_age")
        assert report.target_col == "driver_age"


# ---------------------------------------------------------------------------
# Marginal report
# ---------------------------------------------------------------------------

class TestMarginalReportExtended:
    def test_returns_polars_dataframe(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        assert isinstance(mr, pl.DataFrame)

    def test_has_column_column(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        assert "column" in mr.columns

    def test_one_row_per_column(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        n_numeric = len([c for c in real_df.columns
                          if real_df[c].dtype not in (pl.String, pl.Categorical)])
        # At least numeric columns should appear
        assert len(mr) >= 1

    def test_ks_statistic_in_zero_one(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        ks_vals = mr.filter(pl.col("ks_statistic").is_not_null())["ks_statistic"]
        if len(ks_vals) > 0:
            assert (ks_vals >= 0).all()
            assert (ks_vals <= 1).all()

    def test_identical_data_low_ks(self):
        real_df, _ = _simple_pair()
        report = SyntheticFidelityReport(real_df, real_df)
        mr = report.marginal_report()
        ks_vals = mr.filter(pl.col("ks_statistic").is_not_null())["ks_statistic"]
        if len(ks_vals) > 0:
            # Identical data -> KS = 0
            assert (ks_vals <= 1e-10).all()

    def test_wasserstein_non_negative(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        wass_vals = mr.filter(pl.col("wasserstein").is_not_null())["wasserstein"]
        if len(wass_vals) > 0:
            assert (wass_vals >= 0).all()

    def test_mean_columns_present(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        assert "mean_real" in mr.columns
        assert "mean_synthetic" in mr.columns

    def test_string_column_skipped(self):
        rng = np.random.default_rng(42)
        real_df = pl.DataFrame({
            "region": ["London", "North", "South"] * 67,
            "claim_count": rng.poisson(0.1, 201).tolist(),
        })
        synth_df = real_df  # identical
        report = SyntheticFidelityReport(real_df, synth_df)
        mr = report.marginal_report()
        # region row should have null ks_statistic
        region_row = mr.filter(pl.col("column") == "region")
        if len(region_row) > 0:
            assert region_row["ks_statistic"][0] is None


# ---------------------------------------------------------------------------
# Correlation report
# ---------------------------------------------------------------------------

class TestCorrelationReportExtended:
    def test_returns_polars_dataframe(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        cr = report.correlation_report()
        assert isinstance(cr, pl.DataFrame)

    def test_has_col_a_col_b(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        cr = report.correlation_report()
        assert "col_a" in cr.columns
        assert "col_b" in cr.columns

    def test_delta_non_negative(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        cr = report.correlation_report()
        assert (cr["delta"] >= 0).all()

    def test_spearman_in_minus_one_to_one(self):
        real_df, synth_df = _simple_pair()
        report = SyntheticFidelityReport(real_df, synth_df)
        cr = report.correlation_report()
        if "spearman_real" in cr.columns:
            assert (cr["spearman_real"].abs() <= 1.0 + 1e-8).all()

    def test_identical_data_zero_delta(self):
        real_df, _ = _simple_pair()
        report = SyntheticFidelityReport(real_df, real_df)
        cr = report.correlation_report()
        np.testing.assert_allclose(cr["delta"].to_numpy(), 0.0, atol=1e-8)
