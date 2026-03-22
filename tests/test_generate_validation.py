"""
Regression tests for P1 bug fixes in InsuranceSynthesizer.generate() (batch 3 audit).

Covers:
- generate(0) raises ValueError
- generate(-1) raises ValueError
- generate(1) works correctly
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def fitted_synthesizer():
    """A minimal fitted InsuranceSynthesizer for testing."""
    from insurance_synthetic import InsuranceSynthesizer

    rng = np.random.default_rng(42)
    n = 200
    df = pl.DataFrame({
        "driver_age": rng.integers(20, 70, size=n).tolist(),
        "ncd_years": rng.integers(0, 10, size=n).tolist(),
        "exposure": rng.uniform(0.5, 1.0, size=n).tolist(),
        "claim_count": rng.poisson(0.1, size=n).tolist(),
    })
    synth = InsuranceSynthesizer(method="gaussian", random_state=42)
    synth.fit(df, exposure_col="exposure", frequency_col="claim_count")
    return synth


class TestGenerateValidation:
    """generate() must reject non-positive n."""

    def test_generate_zero_raises_valueerror(self, fitted_synthesizer):
        with pytest.raises(ValueError, match="positive"):
            fitted_synthesizer.generate(0)

    def test_generate_negative_raises_valueerror(self, fitted_synthesizer):
        with pytest.raises(ValueError, match="positive"):
            fitted_synthesizer.generate(-1)

    def test_generate_large_negative_raises_valueerror(self, fitted_synthesizer):
        with pytest.raises(ValueError, match="positive"):
            fitted_synthesizer.generate(-100)

    def test_generate_one_works(self, fitted_synthesizer):
        """generate(1) should return a one-row DataFrame."""
        result = fitted_synthesizer.generate(1)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_generate_positive_n_works(self, fitted_synthesizer):
        """generate(10) should return a 10-row DataFrame."""
        result = fitted_synthesizer.generate(10)
        assert len(result) == 10
