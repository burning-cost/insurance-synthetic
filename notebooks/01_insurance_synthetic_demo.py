# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-synthetic: Vine Copula Portfolio Generator
# MAGIC
# MAGIC This notebook demonstrates the full workflow of the `insurance-synthetic` library:
# MAGIC
# MAGIC 1. Build a realistic UK motor portfolio seed dataset
# MAGIC 2. Fit an `InsuranceSynthesizer` (vine copula + marginals)
# MAGIC 3. Generate 50,000 synthetic policies
# MAGIC 4. Assess fidelity with `SyntheticFidelityReport`
# MAGIC 5. Show that the synthetic data preserves exposure/frequency relationships
# MAGIC
# MAGIC **Why this matters for pricing teams**: you can share the synthetic portfolio
# MAGIC with vendors, use it for model benchmarking, or publish it for reproducible
# MAGIC research — without any real policyholder data leaving the building.

# COMMAND ----------

# MAGIC %pip install insurance-synthetic pyvinecopulib

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_synthetic import (
    InsuranceSynthesizer,
    SyntheticFidelityReport,
    uk_motor_schema,
    fit_marginal,
)

print(f"insurance-synthetic version: {__import__('insurance_synthetic').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a seed portfolio
# MAGIC
# MAGIC In practice you'd load your real data here. We build a 5,000-row synthetic
# MAGIC seed that mimics a mid-market UK motor book: young drivers cluster in London
# MAGIC and South East, NCD correlates with age, claim frequency falls with NCD.

# COMMAND ----------

rng = np.random.default_rng(42)
N_SEED = 5_000

# Driver demographics
driver_age = rng.integers(17, 80, size=N_SEED)

# NCD: correlated with age (can't have more NCD years than driving years)
max_ncd = np.minimum(driver_age - 17, 25)
ncd_raw = max_ncd * rng.beta(2, 1, size=N_SEED)
ncd_years = np.clip(ncd_raw.astype(int), 0, 25)

# Vehicle group: slightly higher for younger drivers (sports cars, cheap hot hatches)
vg_base = 30 - 0.2 * (driver_age - 17) + rng.normal(0, 10, size=N_SEED)
vehicle_group = np.clip(vg_base.astype(int), 1, 50)

# Vehicle age
vehicle_age = rng.integers(0, 20, size=N_SEED)

# Region: London-heavy, realistic UK distribution
region_weights = [0.18, 0.15, 0.10, 0.10, 0.09, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02]
regions = [
    "London", "South East", "East of England", "South West",
    "West Midlands", "East Midlands", "Yorkshire", "North West",
    "North East", "Scotland", "Wales", "Northern Ireland",
]
region = rng.choice(regions, size=N_SEED, p=region_weights)

# Cover type
cover_weights = [0.72, 0.20, 0.08]
cover_type = rng.choice(
    ["Comprehensive", "Third Party Fire & Theft", "Third Party Only"],
    size=N_SEED, p=cover_weights,
)

# Payment method: monthly correlates with risk
payment_weights = [0.55, 0.30, 0.15]
payment_method = rng.choice(
    ["Annual", "Monthly Direct Debit", "Monthly Credit"],
    size=N_SEED, p=payment_weights,
)

# Annual mileage: log-normal
annual_mileage = rng.lognormal(np.log(10_000), 0.5, size=N_SEED).astype(int)
annual_mileage = np.clip(annual_mileage, 1_000, 50_000)

# Exposure: fractional policy years (most are near 1.0)
exposure = rng.beta(8, 1.5, size=N_SEED)
exposure = np.clip(exposure, 0.01, 1.0)

# Frequency: Poisson with exposure offset, declining with NCD
base_freq = 0.08 + 0.003 * vehicle_group - 0.005 * ncd_years
base_freq = np.clip(base_freq, 0.01, 0.6)
claim_count = rng.poisson(base_freq * exposure)
claim_count = np.clip(claim_count, 0, 5)

# Severity: log-normal, zero for no-claim policies
sev_mean = np.log(2000 + 60 * vehicle_group)
claim_amount = np.where(
    claim_count > 0,
    np.maximum(0.0, rng.lognormal(sev_mean, 0.9, size=N_SEED)),
    0.0,
)

seed_df = pl.DataFrame({
    "driver_age": driver_age.tolist(),
    "vehicle_age": vehicle_age.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "region": region.tolist(),
    "ncd_years": ncd_years.tolist(),
    "cover_type": cover_type.tolist(),
    "payment_method": payment_method.tolist(),
    "annual_mileage": annual_mileage.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
    "claim_amount": claim_amount.tolist(),
})

print(f"Seed portfolio: {len(seed_df):,} rows, {len(seed_df.columns)} columns")
print(f"Claim frequency: {seed_df['claim_count'].sum() / seed_df['exposure'].sum():.4f} per policy year")
print(f"Average driver age: {seed_df['driver_age'].mean():.1f}")
print(f"NCD distribution: {seed_df['ncd_years'].describe()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit the synthesiser

# COMMAND ----------

synth = InsuranceSynthesizer(
    method="vine",
    family_set="all",
    trunc_lvl=3,   # Truncate at 3 trees — sufficient for most real portfolios
    n_threads=4,
    random_state=42,
)

synth.fit(
    seed_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
)

synth.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate a synthetic portfolio

# COMMAND ----------

schema = uk_motor_schema()

synthetic_df = synth.generate(
    50_000,
    constraints=schema["constraints"],
    max_resample_attempts=15,
)

print(f"Generated: {len(synthetic_df):,} synthetic policies")
print(f"\nSynthetic claim frequency: {synthetic_df['claim_count'].sum() / synthetic_df['exposure'].sum():.4f} per policy year")
print(f"Real claim frequency:      {seed_df['claim_count'].sum() / seed_df['exposure'].sum():.4f} per policy year")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Marginal comparisons
# MAGIC
# MAGIC Visual check: do the marginal distributions look right?

# COMMAND ----------

# Annualised frequency comparison
real_annual_freq = float(seed_df["claim_count"].sum() / seed_df["exposure"].sum())
synth_annual_freq = float(synthetic_df["claim_count"].sum() / synthetic_df["exposure"].sum())

print("=== Annualised Frequency ===")
print(f"Real:      {real_annual_freq:.4f}")
print(f"Synthetic: {synth_annual_freq:.4f}")
print(f"Ratio:     {synth_annual_freq / real_annual_freq:.4f}  (target: 1.0)")

# Driver age comparison
print("\n=== Driver Age (mean ± std) ===")
print(f"Real:      {seed_df['driver_age'].mean():.1f} ± {seed_df['driver_age'].std():.1f}")
print(f"Synthetic: {synthetic_df['driver_age'].mean():.1f} ± {synthetic_df['driver_age'].std():.1f}")

# NCD distribution
print("\n=== NCD Years (mean ± std) ===")
print(f"Real:      {seed_df['ncd_years'].mean():.1f} ± {seed_df['ncd_years'].std():.1f}")
print(f"Synthetic: {synthetic_df['ncd_years'].mean():.1f} ± {synthetic_df['ncd_years'].std():.1f}")

# Region distribution
print("\n=== Region distribution (top 5) ===")
real_regions = seed_df.group_by("region").agg(pl.len().alias("n")).sort("n", descending=True)
synth_regions = synthetic_df.group_by("region").agg(pl.len().alias("n")).sort("n", descending=True)

real_pct = real_regions.with_columns((pl.col("n") / len(seed_df) * 100).alias("pct"))
synth_pct = synth_regions.with_columns((pl.col("n") / len(synthetic_df) * 100).alias("pct"))

combined = real_pct.rename({"pct": "real_pct", "n": "real_n"}).join(
    synth_pct.rename({"pct": "synth_pct", "n": "synth_n"}),
    on="region",
    how="left",
).head(6)
print(combined)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Fidelity report

# COMMAND ----------

report = SyntheticFidelityReport(
    seed_df,
    synthetic_df,
    exposure_col="exposure",
    target_col="claim_count",
)

print("=== Marginal Fidelity ===")
marg = report.marginal_report()
print(marg.select(["column", "ks_statistic", "ks_pvalue", "wasserstein", "mean_real", "mean_synthetic"]))

# COMMAND ----------

print("=== Correlation Fidelity ===")
corr = report.correlation_report()
frob = float(corr["frobenius_norm"][0])
print(f"Frobenius norm (Spearman matrix distance): {frob:.4f}")
print("\nTop 5 pairs by correlation difference:")
print(corr.head(5).select(["col_a", "col_b", "spearman_real", "spearman_synthetic", "delta"]))

# COMMAND ----------

print("=== Tail Risk ===")
ratio_count = report.tvar_ratio("claim_count", percentile=0.99)
print(f"TVaR ratio (claim_count @ 99th pct): {ratio_count:.4f}  (target: 1.0)")

ew_ks = report.exposure_weighted_ks("driver_age")
print(f"Exposure-weighted KS (driver_age): {ew_ks:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dependency preservation
# MAGIC
# MAGIC The key test for a vine copula synthesiser: does the negative correlation
# MAGIC between NCD years and claim frequency survive synthesis?

# COMMAND ----------

from scipy import stats

for label, df in [("Real", seed_df), ("Synthetic", synthetic_df)]:
    rho, pval = stats.spearmanr(df["ncd_years"].to_numpy(), df["claim_count"].to_numpy())
    print(f"{label}: Spearman rho(NCD, claim_count) = {rho:+.4f}  (p={pval:.4f})")

print()
for label, df in [("Real", seed_df), ("Synthetic", synthetic_df)]:
    rho, pval = stats.spearmanr(df["vehicle_group"].to_numpy(), df["claim_count"].to_numpy())
    print(f"{label}: Spearman rho(vehicle_group, claim_count) = {rho:+.4f}  (p={pval:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Marginal fitting — standalone use
# MAGIC
# MAGIC `fit_marginal` can be used independently of the synthesiser.

# COMMAND ----------

# Fit individual column marginals
severity_marginal = fit_marginal(
    seed_df.filter(pl.col("claim_amount") > 0)["claim_amount"],
    family="auto",
)
print(f"Claim severity: {severity_marginal.family_name()}, AIC={severity_marginal.aic:.1f}")

age_marginal = fit_marginal(seed_df["driver_age"], family="auto")
print(f"Driver age: {age_marginal.family_name()}, AIC={age_marginal.aic:.1f}")

ncd_marginal = fit_marginal(seed_df["ncd_years"], family="auto")
print(f"NCD years: {ncd_marginal.family_name()}, AIC={ncd_marginal.aic:.1f}")

region_marginal = fit_marginal(seed_df["region"])
print(f"Region: {region_marginal.family_name()}")
print(f"  Categories: {region_marginal.categories[:4]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Full markdown fidelity report

# COMMAND ----------

md = report.to_markdown()
print(md)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The vine copula synthesiser:
# MAGIC - Preserves marginal distributions (low KS statistics)
# MAGIC - Preserves pairwise dependencies (NCD negatively correlated with claims)
# MAGIC - Preserves tail risk (TVaR ratio near 1.0)
# MAGIC - Correctly handles exposure-aware frequency generation
# MAGIC - Handles mixed types: continuous, discrete, categorical in the same portfolio
# MAGIC
# MAGIC **What to expect from a good synthesis run**:
# MAGIC - KS statistics < 0.05 for most columns
# MAGIC - Frobenius norm < 2.0 on the Spearman matrix
# MAGIC - TVaR ratio in [0.8, 1.2] for the main target column
# MAGIC - Annualised frequency ratio within 10% of 1.0
# MAGIC
# MAGIC With 5,000 seed rows and 50,000 generated rows, the synthetic portfolio
# MAGIC is large enough to train a frequency GLM or GBM with stable estimates.
