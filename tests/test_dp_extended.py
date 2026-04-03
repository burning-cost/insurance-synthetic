"""Extended tests for DPInsuranceSynthesizer and PrivacyReport.

The heavy fit/generate tests skip if smartnoise-synth is not installed.
PrivacyReport construction tests always run since it's a plain dataclass.
"""

import numpy as np
import polars as pl
import pytest

from insurance_synthetic.dp import DPInsuranceSynthesizer, PrivacyReport


# ---------------------------------------------------------------------------
# PrivacyReport — always available (plain dataclass, no optional deps)
# ---------------------------------------------------------------------------

class TestPrivacyReport:
    def _make_report(self, **kwargs):
        defaults = dict(
            epsilon=1.0,
            epsilon_discretisation=0.1,
            epsilon_synthesis=0.9,
            delta=1e-5,
            mechanism="AIM",
            n_continuous=3,
            n_categorical=2,
            bin_count=20,
            cumulative_epsilon=1.0,
        )
        defaults.update(kwargs)
        return PrivacyReport(**defaults)

    def test_construction(self):
        r = self._make_report()
        assert r.epsilon == pytest.approx(1.0)

    def test_epsilon_discretisation_stored(self):
        r = self._make_report(epsilon_discretisation=0.2)
        assert r.epsilon_discretisation == pytest.approx(0.2)

    def test_mechanism_stored(self):
        r = self._make_report(mechanism="AIM")
        assert r.mechanism == "AIM"

    def test_n_continuous_stored(self):
        r = self._make_report(n_continuous=5)
        assert r.n_continuous == 5

    def test_n_categorical_stored(self):
        r = self._make_report(n_categorical=3)
        assert r.n_categorical == 3

    def test_bin_count_stored(self):
        r = self._make_report(bin_count=10)
        assert r.bin_count == 10

    def test_str_returns_string(self):
        r = self._make_report()
        s = str(r)
        assert isinstance(s, str)
        assert len(s) > 10

    def test_str_contains_epsilon(self):
        r = self._make_report(epsilon=2.5)
        s = str(r)
        assert "2.5" in s or "Epsilon" in s

    def test_str_contains_mechanism(self):
        r = self._make_report()
        s = str(r)
        assert "AIM" in s

    def test_tail_fidelity_default_empty(self):
        r = self._make_report()
        assert r.tail_fidelity == {}

    def test_column_bounds_default_empty(self):
        r = self._make_report()
        assert r.column_bounds == {}

    def test_tail_fidelity_populated(self):
        r = self._make_report()
        r.tail_fidelity["claim_amount_P99"] = 0.65
        assert "claim_amount_P99" in r.tail_fidelity

    def test_warnings_list(self):
        r = self._make_report()
        r.warnings.append("P99 tail degraded.")
        assert len(r.warnings) == 1

    def test_cumulative_epsilon(self):
        r = self._make_report(cumulative_epsilon=1.0)
        assert r.cumulative_epsilon == pytest.approx(1.0)

    def test_delta_stored(self):
        r = self._make_report(delta=1e-6)
        assert r.delta == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# DPInsuranceSynthesizer — construction (no optional deps needed)
# ---------------------------------------------------------------------------

class TestDPSynthesizerConstruction:
    def test_basic_construction(self):
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        assert synth.epsilon == pytest.approx(1.0)

    def test_default_epsilon_positive(self):
        synth = DPInsuranceSynthesizer()
        assert synth.epsilon > 0

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            DPInsuranceSynthesizer(epsilon=-1.0)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            DPInsuranceSynthesizer(epsilon=0.0)

    def test_epsilon_stored(self):
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        assert synth.epsilon == pytest.approx(1.0)

    def test_preprocessor_eps_stored(self):
        synth = DPInsuranceSynthesizer(epsilon=2.0, preprocessor_eps=0.2)
        assert synth.preprocessor_eps == pytest.approx(0.2)

    def test_large_epsilon_accepted(self):
        synth = DPInsuranceSynthesizer(epsilon=10.0)
        assert synth.epsilon == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# DPInsuranceSynthesizer — fit/generate (skip without smartnoise)
# ---------------------------------------------------------------------------

def _make_small_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "driver_age": rng.integers(17, 85, n).tolist(),
        "vehicle_age": rng.integers(0, 20, n).tolist(),
        "region": rng.choice(["London", "North", "South", "East"], n).tolist(),
        "claim_count": rng.poisson(0.12, n).tolist(),
        "exposure": rng.uniform(0.1, 1.0, n).tolist(),
    })


class TestDPSynthesizerFitGenerate:
    @pytest.fixture(autouse=True)
    def skip_if_no_smartnoise(self):
        pytest.importorskip("snsynth", reason="smartnoise-synth not installed")

    def test_fit_and_generate_basic(self):
        df = _make_small_df()
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        synth.fit(
            df,
            categorical_columns=["region"],
            continuous_columns=["driver_age", "vehicle_age", "exposure"],
        )
        out = synth.generate(100)
        assert isinstance(out, pl.DataFrame)
        assert len(out) == 100

    def test_generate_before_fit_raises(self):
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        with pytest.raises((RuntimeError, Exception)):
            synth.generate(10)

    def test_privacy_report_after_fit(self):
        df = _make_small_df()
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        synth.fit(
            df,
            categorical_columns=["region"],
            continuous_columns=["driver_age", "vehicle_age"],
        )
        report = synth.privacy_report()
        assert report is not None

    def test_different_n_produces_different_sizes(self):
        df = _make_small_df()
        synth = DPInsuranceSynthesizer(epsilon=1.0)
        synth.fit(df, categorical_columns=["region"],
                  continuous_columns=["driver_age", "exposure"])
        out50 = synth.generate(50)
        out200 = synth.generate(200)
        assert len(out50) == 50
        assert len(out200) == 200
