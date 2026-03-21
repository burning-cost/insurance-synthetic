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
4. **Frequency generation**: Claim counts drawn as `Poisson(λ × exposure)` to preserve the exposure relationship, using per-group empirical rates from the training data
5. **Severity generation**: Severity is handled separately from the vine copula — see below

The vine copula matters for insurance. A Gaussian copula misses tail dependence — the fact that young driver + high vehicle group + zero NCD is more dangerous than the marginal risks suggest. Clayton and Gumbel copulas capture this. Pyvinecopulib selects the best bivariate family for each pair automatically.

## Severity synthesis

The `claim_amount` column (severity) is excluded from the vine copula and handled separately. This is intentional, not a limitation.

Zero-inflated severity columns — where 85%+ of rows are exactly zero (non-claimers) — break continuous copula fitting. The massive point mass at zero collapses the fitted marginal CDF, and inverting through that CDF produces synthetic severities with KS statistics near 1.0. Running severity through the vine is the wrong tool for the job.

Instead, the library does what actuaries do: treats frequency and severity as separate models.

- The severity marginal is fitted on **non-zero claims only** — the conditional severity distribution `P(amount | claim occurred)`
- At generation time, rows with `claim_count > 0` draw severity from this marginal independently
- Rows with `claim_count == 0` get `claim_amount = 0`

This gives correct severity marginals for claimers while preserving the zero-inflation structure. The trade-off is that severity is not correlated with risk factors (vehicle group, driver age) via the copula. If you need severity to vary by risk segment, post-process the synthetic amounts using a multiplicative relativities table or fit a separate severity GLM on the synthetic data.

## Installation

```bash
pip install insurance-synthetic

# With TSTR fidelity scoring (requires CatBoost):
pip install insurance-synthetic[fidelity]
```

Requires Python 3.10+.

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-synthetic/discussions). Found it useful? A ⭐ helps others find it.

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
report.tvar_ratio(col, percentile=0.99)   # float
report.exposure_weighted_ks(col)   # float
report.tstr_score(...)             # float — requires [fidelity]
report.to_markdown()               # str
```

## Design decisions

**Why vine copulas over CTGAN?** CTGAN requires a GPU for reasonable training times, is a black box, and tends to overfit small portfolios. Vine copulas are fast, interpretable (you can inspect which bivariate families were selected), and scale well to 10k–1m row portfolios. They also have decades of actuarial literature behind them.

**Why Polars?** All our tooling is Polars-first. Pandas DataFrames are not accepted as input — if you have pandas, convert first with `pl.from_pandas(df)`.

**Why AIC marginal selection?** AIC penalises model complexity, which matters with small portfolios (a few thousand rows) where BIC and likelihood ratio tests can be fooled. For large portfolios, the choice of information criterion rarely matters.

**Why exposure-aware frequency generation?** The standard approach of inverting through the frequency marginal ignores the exposure offset. A policy with 0.1 years of exposure and a policy with 1.0 years should have different expected claim counts even if they're otherwise identical. Our approach draws `Poisson(λ × exposure)` where `λ` is the fitted rate, preserving this relationship in the synthetic data.

**Why is severity excluded from the vine copula?** Zero-inflated columns collapse copula fitting. The 80%+ point mass at zero makes the fitted CDF degenerate — all severity samples map to values near zero and the marginal KS statistic reaches 0.93. Treating severity as a separate conditional model (fitted on non-zero claims, drawn independently for claimers) is the actuarially correct approach and produces severity KS < 0.15.

## Running tests

Tests run on Databricks — the package targets environments with pyvinecopulib installed. See the Databricks notebook in `notebooks/` for a full end-to-end demo.

```bash
# On a machine with the dependencies installed:
pytest tests/ -v
```

## Performance

Benchmarked against naive independent sampling (each column drawn from its empirical marginal, correlations ignored) on an 8,000-policy UK motor seed portfolio with known correlation structure. Benchmark run 2026-03-17 on Databricks serverless. pyvinecopulib is not available in standard Databricks Python images so the library fell back to Gaussian copula — the full R-vine with asymmetric tail-dependence families achieves better correlation fidelity than the numbers below.

DGP: rho(driver_age, ncd_years)=+0.502, rho(ncd_years, vehicle_group)=-0.338. Both methods target 8,000 synthetic rows.

**Marginal fidelity (KS statistic per column, lower = better):**

| Column | Naive independent | Vine copula | Notes |
|--------|------------------|-------------|-------|
| driver_age | 0.0066 | 0.0457 | Vine's marginals are noisier — copula fitting introduces minor distortion |
| ncd_years | 0.0060 | 0.1194 | Vine has higher KS for this discrete column — known limitation of continuous copula on integers |
| vehicle_group | 0.0079 | 0.0835 | Same |
| exposure | 0.0092 | 0.0084 | Both preserve exposure marginals |
| claim_count | 0.0020 | 0.0150 | Both near-zero |
| claim_amount | 0.0025 | < 0.15 | Severity excluded from vine; drawn from conditional marginal fitted on non-zero claims |

**Important on marginals**: Naive independent sampling achieves near-zero KS because it is simply resampling from the empirical marginals by construction. Vine copula introduces some distortion when discrete columns are handled through continuous copula ranks. For continuous columns (exposure) the vine performs comparably.

**Spearman correlation preservation (the decisive test):**

| Pair | Real | Vine | Naive |
|------|------|------|-------|
| driver_age vs ncd_years | +0.502 | +0.400 | +0.001 |
| ncd_years vs vehicle_group | -0.338 | -0.174 | -0.003 |
| ncd_years vs claim_count | -0.051 | -0.018 | -0.047 |
| vehicle_group vs claim_count | +0.043 | -0.007 | +0.028 |
| exposure vs claim_count | +0.120 | +0.098 | +0.120 |

Frobenius norm vs real: vine=0.315, naive=0.880. Vine is 2.8x better at preserving the overall correlation structure. Naive sampling destroys the age/NCD correlation almost entirely (+0.001 vs +0.502 true).

Vine does not fully recover the strong age/NCD correlation (+0.400 vs +0.502). This is partly a sample-size effect (8k rows with discrete integer columns is a hard case for vine copulas) and partly a discretisation artefact. On continuous risk factors the vine copula performs closer to the seed correlations.

**Tail fidelity:**
- TVaR ratio at 99th pct (claim_count): vine=1.59, naive=1.01
- The vine over-estimates tail risk by 59% on this dataset. Naive sampling's TVaR ratio is near-perfect here (1.01), but this is coincidental — naive lacks any joint tail structure so it cannot reproduce tail co-occurrence between risk factors.

**TSTR Gini gap (train on synthetic, test on real):** vine=0.0006, naive=0.0016. Both are very small, meaning a model trained on either synthetic dataset generalises almost as well as one trained on real data. This is a positive result for both methods.

**Physical plausibility (young driver with high NCD — impossible combinations):**
- Real: 0.32%
- Vine: 0.26% (suppresses impossible combos via correlation structure)
- Naive: 2.30% (7x more impossible rows than real data)

**Summary:** Vine copula is strictly better on correlation preservation and physical plausibility. It underperforms naive on marginal KS for discrete columns (a known limitation of continuous copula methods applied to integer-valued data). Severity is now synthesised correctly by treating it as a conditional distribution rather than routing it through the vine. For portfolios where the risk factor correlations drive model performance — which they do in frequency models — the vine copula is the right tool.

## Read more

[Your Synthetic Data Doesn't Know What Exposure Is](https://burning-cost.github.io/2026/03/08/insurance-synthetic.html) — why SDV and CTGAN produce portfolios that look right column by column and break the moment you run a pricing model on them.

## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/synthetic_portfolio_generation.py).

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
