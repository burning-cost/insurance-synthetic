"""
Tests for DPInsuranceSynthesizer.

smartnoise-synth is an optional heavy dependency. All tests here mock it out
so the test suite runs in the standard dev environment without the [dp] extra.

Mock strategy:
- We patch snsynth.AIMSynthesizer at the module level using monkeypatch.
- The mock AIM synthesizer stores the fitted DataFrame and returns a sample
  of fixed-value or randomly sampled rows when .sample() is called.
- This tests the DPInsuranceSynthesizer logic (binning, reconstruction,
  budget accounting, report generation) independently of AIM internals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_motor_pd():
    """
    Small pandas DataFrame simulating a UK motor portfolio.

    Using pandas here because DPInsuranceSynthesizer accepts both polars and
    pandas, and we want to test the pandas path specifically without requiring
    polars in every test.
    """
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "driver_age": rng.integers(17, 85, size=n).astype(float),
        "vehicle_age": rng.integers(0, 20, size=n).astype(float),
        "ncd_years": rng.integers(0, 25, size=n).astype(float),
        "exposure": rng.uniform(0.1, 1.0, size=n),
        "region": rng.choice(["London", "South East", "Midlands", "North"], size=n),
        "vehicle_group": rng.integers(1, 50, size=n).astype(str),
    })


@pytest.fixture
def mock_aim_class():
    """
    A mock replacement for snsynth.AIMSynthesizer.

    The mock's sample() method returns a DataFrame of the appropriate shape
    with integer bin labels for continuous columns and the original string
    values for categorical columns. We use bin label 5 (mid-range) to give
    the reconstruction something sensible to work with.
    """
    class _MockAIM:
        def __init__(self, epsilon, delta, verbose=False):
            self.epsilon = epsilon
            self.delta = delta
            self._fitted_columns: list[str] = []
            self._fitted_dtypes: dict[str, str] = {}

        def fit(self, df: pd.DataFrame, preprocessor_eps=0.0, **kwargs):
            self._fitted_columns = list(df.columns)
            for col in df.columns:
                self._fitted_dtypes[col] = str(df[col].dtype)
            self._sample_df = df.reset_index(drop=True)

        def sample(self, n: int) -> pd.DataFrame:
            rng = np.random.default_rng(99)
            rows: dict[str, list] = {}
            for col in self._fitted_columns:
                sample_vals = (
                    self._sample_df[col]
                    .sample(n=n, replace=True, random_state=99)
                    .tolist()
                )
                rows[col] = sample_vals
            return pd.DataFrame(rows)

    return _MockAIM


@pytest.fixture
def patched_synth(mock_aim_class):
    """
    Patch snsynth.AIMSynthesizer and make DPInsuranceSynthesizer importable
    even without smartnoise-synth installed.
    """
    with patch.dict("sys.modules", {"snsynth": MagicMock(AIMSynthesizer=mock_aim_class)}):
        from insurance_synthetic.dp import DPInsuranceSynthesizer
        yield DPInsuranceSynthesizer


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_valid_defaults(self, patched_synth):
        s = patched_synth()
        assert s.epsilon == 1.0
        assert s.delta == 1e-9
        assert s.preprocessor_eps == 0.1
        assert s.bin_count == 20

    def test_negative_epsilon_raises(self, patched_synth):
        with pytest.raises(ValueError, match="epsilon"):
            patched_synth(epsilon=-1.0)

    def test_zero_epsilon_raises(self, patched_synth):
        with pytest.raises(ValueError, match="epsilon"):
            patched_synth(epsilon=0.0)

    def test_preprocessor_eps_out_of_range_raises(self, patched_synth):
        with pytest.raises(ValueError, match="preprocessor_eps"):
            patched_synth(preprocessor_eps=1.0)

    def test_preprocessor_eps_negative_raises(self, patched_synth):
        with pytest.raises(ValueError, match="preprocessor_eps"):
            patched_synth(preprocessor_eps=-0.1)

    def test_bin_count_too_small_raises(self, patched_synth):
        with pytest.raises(ValueError, match="bin_count"):
            patched_synth(bin_count=1)

    def test_custom_epsilon(self, patched_synth):
        s = patched_synth(epsilon=3.0)
        assert s.epsilon == 3.0

    def test_bounds_stored(self, patched_synth):
        bounds = {"driver_age": (17.0, 100.0)}
        s = patched_synth(bounds=bounds)
        assert s.bounds == bounds


# ---------------------------------------------------------------------------
# ImportError when smartnoise-synth is absent
# ---------------------------------------------------------------------------

class TestImportError:
    def test_import_error_on_fit_without_dep(self, small_motor_pd):
        """When snsynth is not available, fit() raises ImportError."""
        # Patch sys.modules so that 'import snsynth' raises ImportError
        import sys
        with patch.dict("sys.modules", {"snsynth": None}):
            # Re-import dp module to get the real (unpatched) class
            if "insurance_synthetic.dp" in sys.modules:
                del sys.modules["insurance_synthetic.dp"]
            from insurance_synthetic.dp import DPInsuranceSynthesizer
            synth = DPInsuranceSynthesizer(epsilon=1.0)
            with pytest.raises(ImportError, match="smartnoise-synth"):
                synth.fit(
                    small_motor_pd,
                    categorical_columns=["region", "vehicle_group"],
                    continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
                )


# ---------------------------------------------------------------------------
# fit() behaviour
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0)
        result = s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
        )
        assert result is s

    def test_fit_sets_is_fitted(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0)
        assert not s._is_fitted
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
        )
        assert s._is_fitted

    def test_fit_stores_column_lists(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
        )
        assert "region" in s._categorical_columns
        assert "vehicle_group" in s._categorical_columns
        assert "driver_age" in s._continuous_columns

    def test_fit_stores_bin_edges(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0, bin_count=10)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
        )
        for col in ["driver_age", "vehicle_age", "ncd_years", "exposure"]:
            assert col in s._bin_edges
            # With bin_count=10, edges array has at most 11 elements
            assert len(s._bin_edges[col]) >= 2

    def test_fit_empty_dataframe_raises(self, patched_synth):
        s = patched_synth(epsilon=1.0)
        empty = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(ValueError, match="empty"):
            s.fit(empty, categorical_columns=["a"], continuous_columns=["b"])

    def test_fit_missing_column_raises(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0)
        with pytest.raises(ValueError):
            s.fit(
                small_motor_pd,
                categorical_columns=["region", "nonexistent_col"],
                continuous_columns=["driver_age"],
            )

    def test_fit_auto_detect_columns(self, patched_synth):
        """Auto-detection: string → categorical, float → continuous."""
        df = pd.DataFrame({
            "region": ["London", "North", "South"] * 10,
            "age": np.random.default_rng(0).uniform(17, 85, 30),
        })
        s = patched_synth(epsilon=1.0)
        s.fit(df)  # no explicit column lists
        assert "region" in s._categorical_columns
        assert "age" in s._continuous_columns

    def test_fit_stores_training_quantiles(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age"],
        )
        assert "driver_age" in s._training_quantiles
        assert "P95" in s._training_quantiles["driver_age"]
        assert "P99" in s._training_quantiles["driver_age"]

    def test_budget_split_is_respected(self, patched_synth, small_motor_pd, mock_aim_class):
        """The AIM synthesizer should receive epsilon * (1 - preprocessor_eps)."""
        s = patched_synth(epsilon=2.0, preprocessor_eps=0.1)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age"],
        )
        # s._synthesizer is the mock instance — epsilon was passed at construction
        assert abs(s._synthesizer.epsilon - 1.8) < 1e-9  # 2.0 * 0.9


# ---------------------------------------------------------------------------
# generate() behaviour
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.fixture(autouse=True)
    def fitted_synth(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=1.0, bin_count=10, random_state=42)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "vehicle_age", "ncd_years", "exposure"],
        )
        self.s = s
        self.training_df = small_motor_pd

    def test_generate_before_fit_raises(self, patched_synth):
        s = patched_synth()
        with pytest.raises(RuntimeError, match="fit"):
            s.generate(100)

    def test_generate_negative_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.s.generate(-1)

    def test_generate_zero_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self.s.generate(0)

    def test_generate_returns_correct_row_count(self):
        for n in [10, 100, 500]:
            out = self.s.generate(n)
            assert len(out) == n, f"Expected {n} rows, got {len(out)}"

    def test_generate_has_all_columns(self):
        out = self.s.generate(50)
        # polars or pandas
        if _HAS_POLARS and hasattr(out, "columns"):
            cols = set(out.columns)
        else:
            cols = set(out.columns)
        expected = set(self.training_df.columns)
        assert cols == expected

    def test_continuous_columns_are_numeric(self):
        out = self.s.generate(50)
        # Convert to pandas for dtype checks if polars
        if hasattr(out, "to_pandas"):
            out_pd = out.to_pandas()
        else:
            out_pd = out
        for col in ["driver_age", "vehicle_age", "ncd_years", "exposure"]:
            assert pd.api.types.is_numeric_dtype(out_pd[col]), (
                f"Column {col} should be numeric after reconstruction"
            )

    def test_continuous_values_within_training_range(self):
        """
        Reconstructed continuous values must lie within [min(training) - slack,
        max(training) + slack]. They won't be exact due to quantile binning but
        should be in the same ballpark.
        """
        out = self.s.generate(200)
        if hasattr(out, "to_pandas"):
            out_pd = out.to_pandas()
        else:
            out_pd = out

        for col in ["driver_age"]:
            train_min = self.training_df[col].min()
            train_max = self.training_df[col].max()
            synth_min = out_pd[col].min()
            synth_max = out_pd[col].max()
            # Allow 10% slack for bin boundary effects
            slack = (train_max - train_min) * 0.15
            assert synth_min >= train_min - slack, (
                f"{col}: synthetic min {synth_min} too far below training min {train_min}"
            )
            assert synth_max <= train_max + slack, (
                f"{col}: synthetic max {synth_max} too far above training max {train_max}"
            )

    def test_sets_last_synthetic(self):
        assert self.s._last_synthetic is None
        self.s.generate(50)
        assert self.s._last_synthetic is not None

    def test_uses_external_bounds(self, patched_synth, small_motor_pd):
        """Externally specified bounds should be used as bin edge limits."""
        s = patched_synth(
            epsilon=1.0,
            bin_count=5,
            bounds={"driver_age": (17.0, 100.0)},
            random_state=0,
        )
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age"],
        )
        edges = s._bin_edges["driver_age"]
        assert edges[0] <= 17.0, "Lower bound from external spec should be at most 17"
        assert edges[-1] >= 100.0, "Upper bound from external spec should be at least 100"


# ---------------------------------------------------------------------------
# privacy_report()
# ---------------------------------------------------------------------------

class TestPrivacyReport:
    @pytest.fixture(autouse=True)
    def fitted_synth(self, patched_synth, small_motor_pd):
        self.s = patched_synth(epsilon=2.0, preprocessor_eps=0.1, bin_count=15, random_state=7)
        self.s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "exposure"],
        )

    def test_report_before_generate(self):
        report = self.s.privacy_report()
        assert report.epsilon == 2.0
        assert abs(report.epsilon_discretisation - 0.2) < 1e-9
        assert abs(report.epsilon_synthesis - 1.8) < 1e-9
        assert report.n_continuous == 2
        assert report.n_categorical == 2
        assert report.tail_fidelity == {}  # no generate() called yet

    def test_report_mechanism_is_aim(self):
        report = self.s.privacy_report()
        assert report.mechanism == "AIM"

    def test_report_cumulative_epsilon(self):
        report = self.s.privacy_report()
        assert report.cumulative_epsilon == 2.0

    def test_report_tail_fidelity_after_generate(self):
        self.s.generate(200)
        report = self.s.privacy_report()
        assert "driver_age_P95" in report.tail_fidelity
        assert "driver_age_P99" in report.tail_fidelity
        assert "exposure_P95" in report.tail_fidelity

    def test_report_tail_fidelity_values_are_positive(self):
        self.s.generate(200)
        report = self.s.privacy_report()
        for key, val in report.tail_fidelity.items():
            assert val > 0, f"Tail fidelity ratio for {key} should be positive, got {val}"

    def test_report_column_bounds_populated(self):
        report = self.s.privacy_report()
        assert "driver_age" in report.column_bounds
        lo, hi = report.column_bounds["driver_age"]
        assert lo < hi

    def test_report_str_renderable(self):
        self.s.generate(100)
        report = self.s.privacy_report()
        rendered = str(report)
        assert "AIM" in rendered
        assert "epsilon" in rendered.lower()
        assert "P95" in rendered
        assert "P99" in rendered

    def test_report_warnings_present(self):
        report = self.s.privacy_report()
        assert len(report.warnings) > 0
        # The tail degradation warning is always present
        tail_warning = any("tail" in w.lower() for w in report.warnings)
        assert tail_warning, "Expected tail degradation warning in privacy report"

    def test_low_epsilon_warning(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=0.3)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age"],
        )
        report = s.privacy_report()
        low_eps_warning = any("very low" in w.lower() for w in report.warnings)
        assert low_eps_warning

    def test_high_epsilon_warning(self, patched_synth, small_motor_pd):
        s = patched_synth(epsilon=15.0)
        s.fit(
            small_motor_pd,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age"],
        )
        report = s.privacy_report()
        high_eps_warning = any("weak" in w.lower() for w in report.warnings)
        assert high_eps_warning


# ---------------------------------------------------------------------------
# uk_motor_dp_bounds and uk_home_dp_bounds
# ---------------------------------------------------------------------------

class TestBoundHelpers:
    def test_uk_motor_bounds_keys(self, patched_synth):
        from insurance_synthetic.dp import uk_motor_dp_bounds
        bounds = uk_motor_dp_bounds()
        assert "driver_age" in bounds
        assert "ncd_years" in bounds
        assert "exposure" in bounds

    def test_uk_motor_bounds_values_valid(self, patched_synth):
        from insurance_synthetic.dp import uk_motor_dp_bounds
        bounds = uk_motor_dp_bounds()
        for col, (lo, hi) in bounds.items():
            assert lo < hi, f"Bound for {col}: lo={lo} must be < hi={hi}"

    def test_uk_home_bounds_keys(self, patched_synth):
        from insurance_synthetic.dp import uk_home_dp_bounds
        bounds = uk_home_dp_bounds()
        assert "buildings_sum_insured" in bounds
        assert "exposure" in bounds

    def test_uk_home_bounds_values_valid(self, patched_synth):
        from insurance_synthetic.dp import uk_home_dp_bounds
        bounds = uk_home_dp_bounds()
        for col, (lo, hi) in bounds.items():
            assert lo < hi


# ---------------------------------------------------------------------------
# _quantile_bin helper
# ---------------------------------------------------------------------------

class TestQuantileBin:
    def test_bin_count(self):
        from insurance_synthetic.dp import _quantile_bin
        arr = np.linspace(0, 100, 500)
        edges, binned = _quantile_bin(arr, n_bins=10)
        # Number of distinct bins used should be <= n_bins
        assert len(np.unique(binned)) <= 10

    def test_all_values_assigned(self):
        from insurance_synthetic.dp import _quantile_bin
        rng = np.random.default_rng(42)
        arr = rng.lognormal(7, 1.5, size=500)
        edges, binned = _quantile_bin(arr, n_bins=20)
        assert len(binned) == len(arr)
        assert np.all(np.isfinite(binned))

    def test_external_bounds_extend_edges(self):
        from insurance_synthetic.dp import _quantile_bin
        arr = np.linspace(20, 80, 200)
        edges, _ = _quantile_bin(arr, n_bins=10, col_bounds=(17.0, 100.0))
        assert edges[0] <= 17.0
        assert edges[-1] >= 100.0

    def test_bin_indices_within_range(self):
        from insurance_synthetic.dp import _quantile_bin
        arr = np.random.default_rng(5).uniform(0, 1000, 300)
        edges, binned = _quantile_bin(arr, n_bins=15)
        assert np.all(binned >= 0)
        assert np.all(binned < len(edges) - 1)

    def test_empty_array(self):
        from insurance_synthetic.dp import _quantile_bin
        edges, binned = _quantile_bin(np.array([]), n_bins=10)
        assert len(binned) == 0

    def test_uniform_array_handles_duplicate_quantiles(self):
        """A constant array has no spread — should not crash."""
        from insurance_synthetic.dp import _quantile_bin
        arr = np.ones(100)
        edges, binned = _quantile_bin(arr, n_bins=10)
        assert len(binned) == 100


# ---------------------------------------------------------------------------
# Polars input path (if polars is available)
# ---------------------------------------------------------------------------

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


@pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")
class TestPolarsInput:
    def test_fit_with_polars_df(self, patched_synth, small_motor_pd):
        pl_df = pl.from_pandas(small_motor_pd)
        s = patched_synth(epsilon=1.0, random_state=0)
        s.fit(
            pl_df,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "exposure"],
        )
        assert s._is_fitted

    def test_generate_returns_polars_df(self, patched_synth, small_motor_pd):
        pl_df = pl.from_pandas(small_motor_pd)
        s = patched_synth(epsilon=1.0, random_state=0)
        s.fit(
            pl_df,
            categorical_columns=["region", "vehicle_group"],
            continuous_columns=["driver_age", "exposure"],
        )
        out = s.generate(50)
        assert isinstance(out, pl.DataFrame)
        assert len(out) == 50
