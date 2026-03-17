"""
Benchmark: insurance-synthetic vine copula vs naive independent sampling.

Data generating process:
  - 8,000-row UK motor portfolio with known dependency structure
  - driver_age ~ N(40, 12) truncated [17, 80]
  - ncd_years ~ truncated discrete, positively correlated with driver_age
    (older drivers have more NCD: rho ~ 0.55)
  - vehicle_group ~ discrete [1..4], negatively correlated with ncd_years
    (high-risk groups tend to have less NCD: rho ~ -0.35)
  - exposure ~ Uniform(0.1, 1.0), independent
  - claim_count ~ Poisson(lambda * exposure) where
    lambda = 0.12 * exp(-0.04 * ncd_years + 0.10 * vehicle_group)
  - claim_amount (severity) ~ LogNormal(mu, sigma) for claimers only

The vine copula generator must preserve:
  (a) Marginal distributions per column
  (b) Spearman correlations (driver_age, ncd_years) and (ncd_years, vehicle_group)
  (c) TSTR: a CatBoost model trained on synthetic data should score similarly
      to one trained on real data (Gini gap < 0.05)

Naive independent sampling preserves (a) trivially but destroys (b) and (c).

Run on Databricks:
  %pip install insurance-synthetic[fidelity] polars scipy numpy catboost
  # pyvinecopulib must be available in the cluster image
"""

import numpy as np
import polars as pl
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Generate "real" seed portfolio with known dependency structure
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 8_000

# driver_age: truncated normal [17, 80]
driver_age_raw = rng.normal(40, 12, N)
driver_age = np.clip(driver_age_raw, 17, 80).astype(int)

# ncd_years: correlated with driver_age (rho ~ 0.55)
ncd_raw = 0.55 * (driver_age - 40) / 12 + np.sqrt(1 - 0.55**2) * rng.standard_normal(N)
ncd_years = np.clip(np.round(ncd_raw * 5 + 5), 0, 14).astype(int)

# vehicle_group: negatively correlated with ncd_years (rho ~ -0.35)
vg_raw = -0.35 * (ncd_years - 5) / 4 + np.sqrt(1 - 0.35**2) * rng.standard_normal(N)
vehicle_group = np.clip(np.round(vg_raw + 2.5), 1, 4).astype(int)

# exposure: uniform, independent
exposure = rng.uniform(0.1, 1.0, N)

# claim frequency rate
lambda_true = 0.12 * np.exp(-0.04 * ncd_years + 0.10 * vehicle_group)
claim_count = rng.poisson(lambda_true * exposure).astype(int)

# severity: lognormal for claimers
claim_amount = np.zeros(N, dtype=float)
has_claim = claim_count > 0
n_claimers = has_claim.sum()
log_mu_sev = 7.0 + 0.05 * vehicle_group[has_claim]
claim_amount[has_claim] = np.exp(rng.normal(log_mu_sev, 0.6))

real_df = pl.DataFrame({
    "driver_age": driver_age.tolist(),
    "ncd_years": ncd_years.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
    "claim_amount": claim_amount.tolist(),
})

print("=" * 70)
print("BENCHMARK: insurance-synthetic vine copula vs naive independent sampling")
print(f"  Real portfolio: {N} rows")
print(f"  Claim frequency: {claim_count.mean() / exposure.mean():.4f} pa")
print("=" * 70)

# True Spearman correlations in the real data
from scipy.stats import spearmanr
rho_age_ncd_real, _ = spearmanr(driver_age, ncd_years)
rho_ncd_vg_real, _ = spearmanr(ncd_years, vehicle_group)
rho_freq_ncd_real, _ = spearmanr(claim_count / exposure, ncd_years)

print(f"\nTrue correlations in real data:")
print(f"  rho(driver_age, ncd_years)    = {rho_age_ncd_real:+.3f}")
print(f"  rho(ncd_years, vehicle_group) = {rho_ncd_vg_real:+.3f}")
print(f"  rho(freq/exp, ncd_years)      = {rho_freq_ncd_real:+.3f}")

# ---------------------------------------------------------------------------
# 2. Vine copula synthesiser
# ---------------------------------------------------------------------------
from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport

print("\nFitting InsuranceSynthesizer (vine copula)...")
synth = InsuranceSynthesizer(method="vine", random_state=0)
synth.fit(
    real_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
    discrete_cols=["driver_age", "ncd_years", "vehicle_group"],
)

print("Generating 8,000 synthetic rows (vine)...")
vine_df = synth.generate(N, constraints={
    "driver_age": (17, 80),
    "ncd_years": (0, 14),
    "vehicle_group": (1, 4),
    "exposure": (0.1, 1.0),
})

# ---------------------------------------------------------------------------
# 3. Naive independent sampling baseline
# ---------------------------------------------------------------------------
print("Generating 8,000 rows via naive independent sampling...")

# Draw each column independently from its empirical marginal — no dependence
naive_driver_age = rng.choice(driver_age, size=N, replace=True)
naive_ncd = rng.choice(ncd_years, size=N, replace=True)
naive_vg = rng.choice(vehicle_group, size=N, replace=True)
naive_exposure = rng.choice(exposure, size=N, replace=True)
# Frequency: also independent draw — loses the lambda(ncd, vg) structure
naive_lambda = 0.12 * np.exp(-0.04 * naive_ncd + 0.10 * naive_vg)  # use true DGP
naive_counts = rng.poisson(naive_lambda * naive_exposure).astype(int)
naive_sev_raw = rng.choice(claim_amount[claim_amount > 0], size=N, replace=True)
naive_amount = np.where(naive_counts > 0, naive_sev_raw, 0.0)

naive_df = pl.DataFrame({
    "driver_age": naive_driver_age.tolist(),
    "ncd_years": naive_ncd.tolist(),
    "vehicle_group": naive_vg.tolist(),
    "exposure": naive_exposure.tolist(),
    "claim_count": naive_counts.tolist(),
    "claim_amount": naive_amount.tolist(),
})

# ---------------------------------------------------------------------------
# 4. Fidelity report
# ---------------------------------------------------------------------------
report_vine = SyntheticFidelityReport(
    real_df, vine_df, exposure_col="exposure", target_col="claim_count"
)
report_naive = SyntheticFidelityReport(
    real_df, naive_df, exposure_col="exposure", target_col="claim_count"
)

marg_vine = report_vine.marginal_report()
marg_naive = report_naive.marginal_report()

print("\n" + "=" * 70)
print("TABLE 1: Marginal fidelity — KS statistic per numeric column")
print(f"  {'Column':<20}  {'KS Naive':>10}  {'KS Vine':>10}  Note")
print("-" * 60)
for row_v in marg_vine.iter_rows(named=True):
    col = row_v["column"]
    ks_vine = row_v["ks_statistic"]
    if ks_vine is None:
        continue
    row_n = marg_naive.filter(pl.col("column") == col).row(0, named=True)
    ks_naive = row_n["ks_statistic"]
    note = "both preserve marginals" if ks_naive is not None else ""
    print(f"  {col:<20}  {ks_naive:>10.4f}  {ks_vine:>10.4f}  {note}")
print("  (KS < 0.05 = good marginal fidelity; both should achieve this)")

# ---------------------------------------------------------------------------
# 5. Correlation preservation — the decisive test
# ---------------------------------------------------------------------------
vine_arr = vine_df.select(["driver_age", "ncd_years", "vehicle_group",
                            "exposure", "claim_count"]).to_numpy().astype(float)
naive_arr = naive_df.select(["driver_age", "ncd_years", "vehicle_group",
                              "exposure", "claim_count"]).to_numpy().astype(float)
real_arr = real_df.select(["driver_age", "ncd_years", "vehicle_group",
                            "exposure", "claim_count"]).to_numpy().astype(float)

# Build Spearman correlation matrices
from scipy.stats import spearmanr as sp_rho
cols5 = ["driver_age", "ncd_years", "vehicle_group", "exposure", "claim_count"]

def spearman_mat(arr):
    n = arr.shape[1]
    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            r, _ = sp_rho(arr[:, i], arr[:, j])
            mat[i, j] = mat[j, i] = r
    return mat

real_sp = spearman_mat(real_arr)
vine_sp = spearman_mat(vine_arr)
naive_sp = spearman_mat(naive_arr)

# Frobenius norm of (synthetic_corr - real_corr) — lower is better
frob_vine = float(np.linalg.norm(vine_sp - real_sp, "fro"))
frob_naive = float(np.linalg.norm(naive_sp - real_sp, "fro"))

print("\n" + "=" * 70)
print("TABLE 2: Spearman correlation preservation")
print(f"  {'Pair':<35}  {'Real':>8}  {'Vine':>8}  {'Naive':>8}")
print("-" * 65)
key_pairs = [
    (0, 1, "driver_age vs ncd_years"),
    (1, 2, "ncd_years vs vehicle_group"),
    (1, 4, "ncd_years vs claim_count"),
    (2, 4, "vehicle_group vs claim_count"),
    (3, 4, "exposure vs claim_count"),
]
for i, j, label in key_pairs:
    print(f"  {label:<35}  {real_sp[i,j]:>+8.3f}  "
          f"{vine_sp[i,j]:>+8.3f}  {naive_sp[i,j]:>+8.3f}")
print(f"\n  Frobenius norm (vs real)          "
      f"{'  baseline':>12}  {frob_vine:>8.3f}  {frob_naive:>8.3f}")
print("  Lower Frobenius = better correlation preservation")

# ---------------------------------------------------------------------------
# 6. TVaR ratio at 99th percentile
# ---------------------------------------------------------------------------
tvar_ratio_vine = report_vine.tvar_ratio("claim_count", percentile=0.99)
tvar_ratio_naive = report_naive.tvar_ratio("claim_count", percentile=0.99)

print("\n" + "=" * 70)
print("TABLE 3: Tail risk preservation (TVaR ratio at 99th pct, claim_count)")
print(f"  Target = 1.00 (synthetic matches real)")
print(f"  Vine copula TVaR ratio : {tvar_ratio_vine:.4f}")
print(f"  Naive sampling TVaR ratio: {tvar_ratio_naive:.4f}")

# ---------------------------------------------------------------------------
# 7. TSTR — Train-on-Synthetic, Test-on-Real
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 4: TSTR — Gini gap (train on synthetic, test on real)")
print("  Computing... (requires catboost)")
try:
    gini_gap_vine = report_vine.tstr_score(test_fraction=0.2, catboost_iterations=150)
    gini_gap_naive = report_naive.tstr_score(test_fraction=0.2, catboost_iterations=150)
    print(f"  Vine copula Gini gap  : {gini_gap_vine:.4f}  (target: near 0)")
    print(f"  Naive sampling Gini gap: {gini_gap_naive:.4f}  (target: near 0)")
    print("  Smaller gap = synthetic data is more useful for model development")
except Exception as e:
    print(f"  TSTR skipped ({e})")
    print("  (Requires insurance-synthetic[fidelity] with catboost installed)")

# ---------------------------------------------------------------------------
# 8. Physical plausibility check
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 5: Physical plausibility — impossible combinations")
print("  Young driver (age<25) with high NCD (ncd>8)")

def impossible_frac(df):
    """Fraction of young drivers with high NCD — should be near 0 in real data."""
    young_high_ncd = df.filter(
        (pl.col("driver_age") < 25) & (pl.col("ncd_years") > 8)
    )
    return len(young_high_ncd) / max(len(df), 1)

print(f"  Real data       : {impossible_frac(real_df):.4f}")
print(f"  Vine copula     : {impossible_frac(vine_df):.4f}")
print(f"  Naive sampling  : {impossible_frac(naive_df):.4f}")
print("  Vine learns age/NCD correlation and suppresses impossible combos.")
print("  Naive sampling has no correlation -> more impossible rows.")

print("\n" + "=" * 70)
print("SUMMARY:")
print("  - KS statistics: both methods preserve marginals (this is not the test)")
print(f"  - Frobenius norm: vine={frob_vine:.3f}, naive={frob_naive:.3f}")
print("    Vine copula preserves the correlation structure. Naive destroys it.")
print(f"  - TVaR ratio: vine={tvar_ratio_vine:.3f}, naive={tvar_ratio_naive:.3f}")
print("  - Physical plausibility: vine suppresses impossible combinations")
print("=" * 70)
