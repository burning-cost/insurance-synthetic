"""
Shared fixtures for the insurance-synthetic test suite.

We use synthetic data generated from known distributions for most tests.
Where possible, expected values are derived analytically or from well-known
reference implementations so the tests are not merely self-consistent.

Seed 42 is used throughout for reproducibility. All DataFrames use Polars.
"""

import numpy as np
import polars as pl
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Random number generator — fixed seed for reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Small motor portfolio (100 rows) — used for quick unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def small_motor_df(rng):
    """
    A 100-row synthetic motor portfolio.

    Distributions are chosen to be clearly separable so that marginal
    recovery tests have power.
    """
    n = 100
    return pl.DataFrame({
        "driver_age": rng.integers(17, 85, size=n).tolist(),
        "vehicle_age": rng.integers(0, 20, size=n).tolist(),
        "vehicle_group": rng.integers(1, 50, size=n).tolist(),
        "region": rng.choice(
            ["London", "South East", "Midlands", "North", "Scotland"],
            size=n,
        ).tolist(),
        "ncd_years": rng.integers(0, 20, size=n).tolist(),
        "exposure": rng.uniform(0.1, 1.0, size=n).tolist(),
        "claim_count": rng.poisson(0.12, size=n).tolist(),
    })


# ---------------------------------------------------------------------------
# Medium motor portfolio (1000 rows) — used for copula and fidelity tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def medium_motor_df(rng):
    """
    A 1000-row portfolio with realistic correlations:
    - Young drivers tend to have low NCD
    - High vehicle group correlates with younger drivers (sports cars)
    - Frequency declines with NCD (the whole point of NCD)
    """
    n = 1000

    # Driver age: skewed young (lots of new policyholders)
    driver_age = rng.integers(17, 75, size=n)

    # NCD: positively correlated with driver age, but capped at 25
    ncd_base = (driver_age - 17) * 0.8 + rng.normal(0, 3, size=n)
    ncd_years = np.clip(ncd_base, 0, 25).astype(int)

    # Vehicle group: slightly higher for younger drivers
    vg_base = 25 - (driver_age - 17) * 0.1 + rng.normal(0, 8, size=n)
    vehicle_group = np.clip(vg_base, 1, 50).astype(int)

    # Exposure: uniform
    exposure = rng.uniform(0.1, 1.0, size=n)

    # Frequency rate: decreasing with NCD, increasing with vehicle group
    lambda_base = 0.08 + 0.003 * vehicle_group - 0.005 * ncd_years
    lambda_adj = np.clip(lambda_base, 0.01, 0.5)
    claim_count = rng.poisson(lambda_adj * exposure)

    # Severity: log-normal, higher for high vehicle group
    sev_mean = 2000 + 50 * vehicle_group
    claim_amount = np.where(
        claim_count > 0,
        np.maximum(0, rng.lognormal(np.log(sev_mean), 0.8, size=n)),
        0.0,
    )

    return pl.DataFrame({
        "driver_age": driver_age.tolist(),
        "vehicle_age": rng.integers(0, 15, size=n).tolist(),
        "vehicle_group": vehicle_group.tolist(),
        "region": rng.choice(
            ["London", "South East", "Midlands", "North", "Scotland",
             "Wales", "East", "North West"],
            size=n,
        ).tolist(),
        "ncd_years": ncd_years.tolist(),
        "exposure": exposure.tolist(),
        "claim_count": claim_count.tolist(),
        "claim_amount": claim_amount.tolist(),
    })


# ---------------------------------------------------------------------------
# Known-distribution series for marginal recovery tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gamma_series(rng):
    """200 samples from Gamma(a=2, scale=500) — property claim severity."""
    data = rng.gamma(2.0, 500.0, size=200)
    return pl.Series("claim_severity", data.tolist())


@pytest.fixture(scope="session")
def lognormal_series(rng):
    """200 samples from LogNormal(mean=7.5, sigma=1.0)."""
    data = rng.lognormal(7.5, 1.0, size=200)
    return pl.Series("claim_amount", data.tolist())


@pytest.fixture(scope="session")
def poisson_series(rng):
    """500 samples from Poisson(mu=0.15) — claim counts."""
    data = rng.poisson(0.15, size=500)
    return pl.Series("claim_count", data.tolist())


@pytest.fixture(scope="session")
def normal_series(rng):
    """300 samples from Normal(mu=45, sigma=12) — driver age."""
    data = np.clip(rng.normal(45, 12, size=300), 17, 90)
    return pl.Series("driver_age", data.tolist())


@pytest.fixture(scope="session")
def categorical_series(rng):
    """200 samples from a 5-category distribution."""
    cats = rng.choice(["London", "South East", "Midlands", "North", "Scotland"], size=200)
    return pl.Series("region", cats.tolist())
