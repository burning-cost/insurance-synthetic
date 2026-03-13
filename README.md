# insurance-synthetic

[![Tests](https://github.com/burning-cost/insurance-synthetic/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-synthetic/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-synthetic)](https://pypi.org/project/insurance-synthetic/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Generate synthetic insurance portfolio data using vine copulas.

## The problem

UK pricing teams frequently need realistic insurance data they cannot actually share:

- Vendor demos require a motor portfolio with the right marginals and correlations, but you can't hand over policyholder data
- Model benchmarking across teams needs a common dataset that doesn't exist
- Privacy regulations mean actuarial science students and researchers rarely see real claims data

Generic synthetic data tools (SDV, CTGAN, TVAE) generate plausible-looking rows, but they don't understand insurance structure. They produce synthetic portfolios where claim counts are independent of exposure, young drivers don't correlate with zero NCD, and severity distributions have the wrong tail shape. A model trained on that synthetic data won't generalise to real portfolios.

This library solves that.

## What it does

`insurance-synthetic` generates synthetic portfolios using R-vine copulas (via [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)):

1. **Marginal fitting**: Each column gets the best-fitting marginal by AIC — Gamma, LogNormal, Poisson, NegBin, Normal, Beta, or categorical encoding
2. **PIT transform**: Every column is mapped to uniform [0,1] via its CDF
3. **Vine copula**: Pairwise dependencies (including tail dependence) are captured by a fitted R-vine
4. **Generation**: Sample from the vine, invert through marginals, then regenerate frequency as `Poisson(λ × exposure)` to preserve the exposure relationship

The vine copula matters for insurance. A Gaussian copula misses tail dependence — the fact that young driver + high vehicle group + zero NCD is more dangerous than the marginal risks suggest. Clayton and Gumbel copulas capture this. Pyvinecopulib selects the best bivariate family for each pair automatically.

## Installation

```bash
pip install insurance-synthetic

# With TSTR fidelity scoring (requires CatBoost):
pip install insurance-synthetic[fidelity]
```

Requires Python 3.10+.

## Quick start

```python
import numpy as np
import polars as pl
from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport

# Seed data: either load from insurance-datasets or generate minimal inline data.
#
# Option A — use the published synthetic seed portfolio (recommended):
#   uv add insurance-datasets
#   from insurance_datasets import load_motor
#   real_df = load_motor()  # 50,000-row UK motor portfolio with known DGP
#
# Option B — minimal inline portfolio for a quick demo:
rng = np.random.default_rng(42)
n = 5_000
real_df = pl.DataFrame({
    'driver_age':    rng.integers(17, 75, size=n).tolist(),
    'vehicle_group': rng.integers(1, 20, size=n).tolist(),
    'ncd_years':     rng.integers(0, 15, size=n).tolist(),
    'region':        rng.choice(['London', 'South East', 'North West', 'Scotland'], size=n).tolist(),
    'exposure':      rng.uniform(0.1, 1.0, size=n).tolist(),
    'claim_count':   rng.poisson(0.07, size=n).tolist(),
    'claim_amount':  (rng.gamma(2.0, scale=1500, size=n) * (rng.poisson(0.07, size=n) > 0)).tolist(),
})

# Fit on your portfolio (real or synthetic seed above)
synth = InsuranceSynthesizer(random_state=42)
synth.fit(
    real_df,
    exposure_col='exposure',
    frequency_col='claim_count',
    severity_col='claim_amount',
)
synth.summary()

# Generate 50,000 synthetic policies
synthetic_df = synth.generate(50_000, constraints={
    'driver_age': (17, 90),
    'ncd_years': (0, 25),
    'exposure': (0.01, 1.0),
})

# Measure fidelity
report = SyntheticFidelityReport(
    real_df, synthetic_df,
    exposure_col='exposure',
    target_col='claim_count',
)
print(report.to_markdown())
```

## UK motor schema

The library ships a pre-built column specification for a UK private motor portfolio:

```python
from insurance_synthetic import uk_motor_schema

schema = uk_motor_schema()
# {
#   'columns': [ColumnSpec(name='driver_age', dtype='int', min_val=17, max_val=90), ...],
#   'constraints': {'driver_age': (17, 90), 'exposure': (0.01, 1.0), ...},
#   'description': 'UK private motor portfolio schema. ...'
# }
```

Columns: `driver_age`, `vehicle_age`, `vehicle_group`, `region`, `ncd_years`, `cover_type`, `payment_method`, `annual_mileage`, `exposure`, `claim_count`, `claim_amount`.

## Fidelity metrics

`SyntheticFidelityReport` measures synthesis quality at three levels:

| Metric | What it checks | Target |
|--------|---------------|--------|
| KS statistic | Marginal distribution per column | < 0.05 is excellent |
| Wasserstein distance | Marginal shape (normalised by std) | < 0.1 is good |
| Spearman Frobenius | Correlation matrix distance | Low |
| TVaR ratio | Tail risk preservation at 99th pct | ≈ 1.0 |
| Exposure-weighted KS | Marginal fidelity weighted by policy year | < 0.05 is excellent |
| TSTR Gini gap | Train-on-Synthetic, Test-on-Real | ≈ 0.0 |

The TSTR Gini gap is the most demanding test: if a CatBoost model trained on synthetic data scores within a small margin of one trained on real data, the synthetic portfolio is genuinely useful for pricing model development.

```python
# Requires insurance-synthetic[fidelity]
gini_gap = report.tstr_score(test_fraction=0.2, catboost_iterations=200)
print(f"TSTR Gini gap: {gini_gap:.4f}")  # target: near 0
```

## API reference

### `InsuranceSynthesizer`

```python
InsuranceSynthesizer(
    method='vine',          # 'vine' | 'gaussian'
    marginals='auto',       # 'auto' | dict of column -> scipy family name
    family_set='all',       # pyvinecopulib family set
    trunc_lvl=None,         # vine truncation level (None = full)
    n_threads=1,
    random_state=None,
)
```

`.fit(df, exposure_col, frequency_col, severity_col, categorical_cols, discrete_cols)`
`.generate(n, constraints, max_resample_attempts)` → `pl.DataFrame`
`.summary()` → `str`
`.get_params()` → `dict`

### `fit_marginal`

Standalone function for fitting a single column:

```python
from insurance_synthetic import fit_marginal
m = fit_marginal(series, family='auto')  # or 'gamma', 'lognorm', 'norm', etc.
m.cdf(values)   # → np.ndarray of probabilities
m.ppf(probs)    # → np.ndarray of values
m.rvs(100)      # → np.ndarray of random samples
m.family_name() # → 'gamma', 'lognorm', etc.
m.aic           # → float
```

### `SyntheticFidelityReport`

```python
report = SyntheticFidelityReport(real_df, synthetic_df, exposure_col, target_col)
report.marginal_report()            # pl.DataFrame — KS, Wasserstein per column
report.correlation_report()        # pl.DataFrame — Spearman comparison
report.tvar_ratio(col, pct=0.99)   # float
report.exposure_weighted_ks(col)   # float
report.tstr_score(...)             # float — requires [fidelity]
report.to_markdown()               # str
```

## Design decisions

**Why vine copulas over CTGAN?** CTGAN requires a GPU for reasonable training times, is a black box, and tends to overfit small portfolios. Vine copulas are fast, interpretable (you can inspect which bivariate families were selected), and scale well to 10k–1m row portfolios. They also have decades of actuarial literature behind them.

**Why Polars?** All our tooling is Polars-first. Pandas DataFrames are not accepted as input — if you have pandas, convert first with `pl.from_pandas(df)`.

**Why AIC marginal selection?** AIC penalises model complexity, which matters with small portfolios (a few thousand rows) where BIC and likelihood ratio tests can be fooled. For large portfolios, the choice of information criterion rarely matters.

**Why exposure-aware frequency generation?** The standard approach of inverting through the frequency marginal ignores the exposure offset. A policy with 0.1 years of exposure and a policy with 1.0 years should have different expected claim counts even if they're otherwise identical. Our approach draws `Poisson(λ × exposure)` where `λ` is the fitted rate, preserving this relationship in the synthetic data.

## Running tests

Tests run on Databricks — the package targets environments with pyvinecopulib installed. See the Databricks notebook in `notebooks/` for a full end-to-end demo.

```bash
# On a machine with the dependencies installed:
pytest tests/ -v
```

## Read more

[Your Synthetic Data Doesn't Know What Exposure Is](https://burning-cost.github.io/2026/03/08/insurance-synthetic.html) — why SDV and CTGAN produce portfolios that look right column by column and break the moment you run a pricing model on them.

## Capabilities

The notebook at `notebooks/01_insurance_synthetic_demo.py` fits an `InsuranceSynthesizer` on a 5,000-row UK motor seed portfolio and generates 50,000 synthetic policies. It demonstrates:

- **Marginal fidelity**: KS statistics for most columns land below 0.05, confirming that the per-column distributions are closely reproduced by the vine copula.
- **Dependency preservation**: Spearman rank correlation between NCD years and claim count is negative in both real and synthetic data, and matches in magnitude — the vine captures this structural relationship rather than treating columns as independent.
- **Annualised frequency stability**: The claim frequency per policy year in the synthetic portfolio matches the seed within a few percent, confirming the exposure-aware generation step works correctly.
- **Tail risk**: TVaR ratio at the 99th percentile stays near 1.0, indicating that the tail of the claim count distribution is preserved — important for catastrophe loading and reserving applications.
- **Mixed types handled**: The same synthesiser handles continuous (mileage, severity), discrete (claim count, NCD), and categorical (region, cover type) columns within a single vine copula fit.

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Fixed synthetic datasets with published DGPs — use when you need reproducible benchmarks rather than portfolio-fitted synthesis |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation — synthetic data can be used to stress-test CV strategies before applying to real books |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | GLM interaction detection — synthetic portfolios with known interaction structure are useful for validating the CANN pipeline |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — generate synthetic portfolios to test fairness tooling without exposing real policyholder data |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Licence

MIT. See [LICENSE](LICENSE).
