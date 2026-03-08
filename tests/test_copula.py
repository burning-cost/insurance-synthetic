"""
Tests for the vine copula wrapper (_copula.py).

We check:
- VineCopulaModel fits on known correlated data
- Simulated samples are in [0,1]
- The fitted copula reproduces the Spearman correlation approximately
- Gaussian fallback mode works when pyvinecopulib is unavailable (we can
  test this by explicitly setting _using_fallback=True)
- Simulation before fit raises
"""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from insurance_synthetic._copula import VineCopulaModel


# ---------------------------------------------------------------------------
# Basic vine copula tests
# ---------------------------------------------------------------------------

class TestVineCopulaModel:
    @pytest.fixture
    def correlated_uniforms(self):
        """
        100 rows of 3-dimensional data with strong pairwise correlations.
        Col 0 and 1 are positively correlated (rho~0.8).
        Col 1 and 2 are negatively correlated (rho~-0.5).
        """
        rng = np.random.default_rng(42)
        z1 = rng.standard_normal(100)
        z2 = 0.8 * z1 + 0.6 * rng.standard_normal(100)
        z3 = -0.5 * z1 + np.sqrt(1 - 0.25) * rng.standard_normal(100)
        from scipy.stats import norm
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)
        u3 = norm.cdf(z3)
        return np.column_stack([u1, u2, u3])

    def test_fit_returns_self(self, correlated_uniforms):
        model = VineCopulaModel()
        result = model.fit(correlated_uniforms)
        assert result is model

    def test_simulate_returns_correct_shape(self, correlated_uniforms):
        model = VineCopulaModel()
        model.fit(correlated_uniforms)
        u = model.simulate(200)
        assert u.shape == (200, 3)

    def test_simulated_values_in_unit_interval(self, correlated_uniforms):
        model = VineCopulaModel()
        model.fit(correlated_uniforms)
        u = model.simulate(500)
        assert u.min() >= 0.0
        assert u.max() <= 1.0

    def test_simulate_before_fit_raises(self):
        model = VineCopulaModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.simulate(10)

    def test_spearman_correlation_approximately_recovered(self, correlated_uniforms):
        """
        The vine should capture the correlation structure well enough that
        simulated Spearman rho is within 0.3 of the original.
        """
        # True Spearman from the input
        real_rho_01, _ = stats.spearmanr(correlated_uniforms[:, 0], correlated_uniforms[:, 1])
        real_rho_12, _ = stats.spearmanr(correlated_uniforms[:, 1], correlated_uniforms[:, 2])

        model = VineCopulaModel()
        model.fit(correlated_uniforms)
        u_sim = model.simulate(2000, rng=np.random.default_rng(0))

        sim_rho_01, _ = stats.spearmanr(u_sim[:, 0], u_sim[:, 1])
        sim_rho_12, _ = stats.spearmanr(u_sim[:, 1], u_sim[:, 2])

        assert abs(sim_rho_01 - real_rho_01) < 0.3, \
            f"Spearman rho(0,1): real={real_rho_01:.3f}, simulated={sim_rho_01:.3f}"
        assert abs(sim_rho_12 - real_rho_12) < 0.3, \
            f"Spearman rho(1,2): real={real_rho_12:.3f}, simulated={sim_rho_12:.3f}"

    def test_seed_reproducibility(self, correlated_uniforms):
        model = VineCopulaModel()
        model.fit(correlated_uniforms)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        u1 = model.simulate(50, rng=rng1)
        u2 = model.simulate(50, rng=rng2)
        np.testing.assert_array_almost_equal(u1, u2)

    def test_summary_returns_string(self, correlated_uniforms):
        model = VineCopulaModel()
        model.fit(correlated_uniforms)
        s = model.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_unfitted_summary_string(self):
        model = VineCopulaModel()
        s = model.summary()
        assert "not fitted" in s.lower()


# ---------------------------------------------------------------------------
# Gaussian fallback tests (force _using_fallback = True)
# ---------------------------------------------------------------------------

class TestGaussianFallback:
    @pytest.fixture
    def fallback_model(self):
        """VineCopulaModel with fallback forced on."""
        model = VineCopulaModel()
        model._using_fallback = True
        return model

    def test_fallback_fit_and_simulate(self, fallback_model):
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, (100, 4))
        fallback_model.fit(u)
        samples = fallback_model.simulate(200, rng=np.random.default_rng(0))
        assert samples.shape == (200, 4)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_fallback_preserves_correlation_direction(self, fallback_model):
        """Gaussian fallback should still reproduce positive vs negative correlations."""
        rng_fit = np.random.default_rng(42)
        z = rng_fit.standard_normal((300, 2))
        # Strong positive correlation
        z[:, 1] = 0.9 * z[:, 0] + 0.1 * rng_fit.standard_normal(300)
        from scipy.stats import norm
        u = norm.cdf(z)
        fallback_model.fit(u)
        u_sim = fallback_model.simulate(1000, rng=np.random.default_rng(1))
        rho, _ = stats.spearmanr(u_sim[:, 0], u_sim[:, 1])
        assert rho > 0.5, f"Expected positive correlation, got {rho:.3f}"

    def test_fallback_property(self, fallback_model):
        assert fallback_model.using_fallback is True


# ---------------------------------------------------------------------------
# High-dimensional test (more columns than typical vine tests)
# ---------------------------------------------------------------------------

class TestHighDimensional:
    def test_fit_simulate_8d(self):
        """Vine copula should handle 8 variables without crashing."""
        rng = np.random.default_rng(42)
        u = rng.uniform(0.01, 0.99, (300, 8))
        model = VineCopulaModel(trunc_lvl=2)  # truncate for speed
        model.fit(u)
        out = model.simulate(100)
        assert out.shape == (100, 8)
        assert np.all(out >= 0)
        assert np.all(out <= 1)
