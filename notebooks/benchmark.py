"""
Benchmark: VineCopulaSynthesizer vs independent column sampling
===============================================================

Demonstrates the core claim of insurance-synthetic: vine copulas preserve
multivariate structure that naive independent sampling destroys.

The naive baseline shuffles each column independently. This preserves marginal
distributions perfectly but destroys all between-column dependencies. The vine
synthesiser preserves both.

Why this matters for insurance pricing:
- A pricing model trained on synthetic data where young drivers are independent
  of zero NCD will underfit the interaction in real data
- Reserve calculations depend on the joint tail behaviour of severity and
  frequency — independent sampling produces the wrong joint tail
- Any downstream model comparison (TSTR) will show the vine-synthesised data
  is materially more useful than the independently sampled version

Metrics:
1. Frobenius norm of Spearman correlation matrix difference
   (vine should be much lower than independent)
2. Marginal KS test p-values per column
   (both should be high — neither should distort marginals much)
3. Pairwise mutual information for selected pairs
   (vine should be close to real; independent should be near-zero)
4. TSTR Gini gap (train on synthetic, test on real)
   (vine should be closer to 0 than independent)

Run on Databricks (requires pyvinecopulib, catboost, polars, scipy).
"""

import time
import warnings

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Naive baseline: independent column shuffling
# ---------------------------------------------------------------------------

def independent_shuffle(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """
    Generate synthetic data by shuffling each column independently.

    This preserves marginal distributions exactly but destroys all
    between-column dependencies. It is the simplest possible synthetic
    data approach and the natural baseline for vine copula synthesis.
    """
    rng = np.random.default_rng(seed)
    shuffled = {}
    for col in df.columns:
        vals = df[col].to_numpy().copy()
        rng.shuffle(vals)
        shuffled[col] = vals
    return pl.DataFrame(shuffled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman_frobenius(real: pl.DataFrame, synth: pl.DataFrame) -> float:
    """
    Frobenius norm of the difference between Spearman correlation matrices.

    Lower is better. The independent baseline will have near-zero off-diagonal
    entries so the Frobenius norm will equal the Frobenius norm of the real
    correlation matrix itself.
    """
    numeric_cols = [
        c for c in real.columns
        if c in synth.columns
        and real[c].dtype not in (pl.Utf8, pl.Categorical, pl.String)
    ]
    real_arrays = np.column_stack([real[c].to_numpy().astype(float) for c in numeric_cols])
    synth_arrays = np.column_stack([synth[c].to_numpy().astype(float) for c in numeric_cols])

    real_corr = stats.spearmanr(real_arrays).statistic
    synth_corr = stats.spearmanr(synth_arrays).statistic

    if real_corr.ndim == 0:  # 2-column edge case
        real_corr = np.array([[1.0, float(real_corr)], [float(real_corr), 1.0]])
        synth_corr = np.array([[1.0, float(synth_corr)], [float(synth_corr), 1.0]])

    return float(np.linalg.norm(real_corr - synth_corr, "fro"))


def marginal_ks_pvalues(real: pl.DataFrame, synth: pl.DataFrame) -> dict[str, float]:
    """KS test p-values per column. High p-value = marginals match."""
    result = {}
    for col in real.columns:
        if col not in synth.columns:
            continue
        if real[col].dtype in (pl.Utf8, pl.Categorical, pl.String):
            continue
        r = real[col].drop_nulls().to_numpy().astype(float)
        s = synth[col].drop_nulls().to_numpy().astype(float)
        _, pval = ks_2samp(r, s)
        result[col] = float(pval)
    return result


def mutual_info_approx(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Approximate mutual information via histogram binning.

    This is not the most accurate estimator but is fast, sufficient for
    comparing vine vs independent synthesis qualitatively.
    """
    x_scaled = np.clip((x - x.min()) / (x.max() - x.min() + 1e-10), 0, 1 - 1e-10)
    y_scaled = np.clip((y - y.min()) / (y.max() - y.min() + 1e-10), 0, 1 - 1e-10)

    x_bins = (x_scaled * bins).astype(int)
    y_bins = (y_scaled * bins).astype(int)

    joint_counts = np.zeros((bins, bins))
    for xi, yi in zip(x_bins, y_bins):
        joint_counts[xi, yi] += 1

    joint_prob = joint_counts / joint_counts.sum()
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
    return float(mi)


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised Gini (2*AUC - 1) for regression."""
    n = len(y_true)
    if n == 0 or np.sum(y_true) == 0:
        return 0.0
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order]
    cumsum = np.cumsum(y_sorted)
    gini = (np.sum(cumsum) / np.sum(y_true) - (n + 1) / 2) / n
    return float(gini * 2)


def tstr_gini_gap(
    real_df: pl.DataFrame,
    synth_df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, float]:
    """
    Train-on-Synthetic/Train-on-Real, Test-on-Real Gini comparison.

    Returns real_gini, synth_gini, and the gap (real - synth).
    Gap near 0 means the synthetic data trains a model almost as good
    as the real data.
    """
    from catboost import CatBoostRegressor

    rng = np.random.default_rng(seed)
    n = len(real_df)
    test_idx = rng.choice(n, size=int(n * test_fraction), replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    real_train = real_df[train_idx.tolist()]
    real_test = real_df[test_idx.tolist()]

    cat_features = [c for c in feature_cols if real_df[c].dtype in (pl.Utf8, pl.Categorical, pl.String)]
    cat_idx = [feature_cols.index(c) for c in cat_features]

    x_real = real_train.select(feature_cols).to_pandas()
    y_real = real_train[target_col].to_numpy()
    x_synth = synth_df.select(feature_cols).to_pandas()
    y_synth = synth_df[target_col].to_numpy()
    x_test = real_test.select(feature_cols).to_pandas()
    y_test = real_test[target_col].to_numpy()

    for col in cat_features:
        x_real[col] = x_real[col].astype(str)
        x_synth[col] = x_synth[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    params = {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 4,
        "loss_function": "Poisson",
        "verbose": False,
        "random_seed": seed,
        "allow_writing_files": False,
    }

    model_real = CatBoostRegressor(**params)
    model_real.fit(x_real, y_real, cat_features=cat_idx)

    model_synth = CatBoostRegressor(**params)
    model_synth.fit(x_synth, y_synth, cat_features=cat_idx)

    real_gini = gini_coefficient(y_test, model_real.predict(x_test))
    synth_gini = gini_coefficient(y_test, model_synth.predict(x_test))

    return {
        "real_gini": real_gini,
        "synth_gini": synth_gini,
        "gap": real_gini - synth_gini,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 65)
    print("insurance-synthetic  benchmark")
    print("VineCopulaSynthesizer vs independent column shuffling")
    print("=" * 65)
    print()

    from insurance_synthetic import InsuranceSynthesizer

    # Seed portfolio: generate structured data with known correlations
    print("Generating seed portfolio (5,000 policies, structured DGP)...")
    rng = np.random.default_rng(0)
    n = 5_000

    # Structured generation: variables are NOT independent
    # ncd_years and driver_age are correlated (older drivers have more NCD)
    # vehicle_group and claim_count are correlated
    # claim_amount and vehicle_group are correlated
    driver_age = rng.integers(17, 80, size=n)

    # NCD: correlated with age (older drivers more likely to have NCD)
    max_ncd = np.clip((driver_age - 17) // 3, 0, 9)
    ncd_years = np.array([rng.integers(0, max(1, m + 1)) for m in max_ncd])

    vehicle_group = np.clip(rng.normal(25, 12, size=n).astype(int), 1, 50)
    annual_mileage = rng.lognormal(9.0, 0.5, size=n).astype(int)
    exposure = rng.uniform(0.1, 1.0, size=n)

    # Frequency depends on age, NCD, vehicle_group
    log_lambda = (
        -3.0
        + 0.025 * vehicle_group
        - 0.08 * ncd_years
        + 0.45 * (driver_age < 25).astype(float)
    )
    claim_count = rng.poisson(exposure * np.exp(log_lambda))

    # Severity depends on vehicle_group
    log_mu = 7.5 + 0.02 * vehicle_group
    claim_amount = np.where(
        claim_count > 0,
        rng.gamma(2.0, np.exp(log_mu) / 2.0, size=n),
        0.0,
    )

    region = rng.choice(["London", "South East", "North West", "Scotland", "Midlands"], size=n)

    real_df = pl.DataFrame({
        "driver_age":    driver_age.tolist(),
        "vehicle_group": vehicle_group.tolist(),
        "ncd_years":     ncd_years.tolist(),
        "annual_mileage": annual_mileage.tolist(),
        "region":        region.tolist(),
        "exposure":      exposure.tolist(),
        "claim_count":   claim_count.tolist(),
        "claim_amount":  claim_amount.tolist(),
    })

    print(f"  Seed portfolio: {real_df.shape[0]:,} rows, {real_df.shape[1]} columns")
    print(f"  Claim frequency: {claim_count.sum() / exposure.sum():.4f} per policy year")
    print(f"  True Spearman(ncd_years, driver_age): {stats.spearmanr(ncd_years, driver_age).statistic:.3f}")
    print(f"  True Spearman(vehicle_group, claim_count): {stats.spearmanr(vehicle_group, claim_count).statistic:.3f}")
    print()

    # -----------------------------------------------------------------------
    # Baseline: independent column shuffling
    # -----------------------------------------------------------------------
    print("Generating baseline: independent column shuffle...")
    t0 = time.time()
    indep_df = independent_shuffle(real_df, seed=42)
    t_indep = time.time() - t0
    print(f"  Done in {t_indep:.2f}s")
    print()

    # -----------------------------------------------------------------------
    # Vine copula synthesis
    # -----------------------------------------------------------------------
    print("Fitting VineCopulaSynthesizer...")
    t0 = time.time()
    synth = InsuranceSynthesizer(
        method="vine",
        marginals="auto",
        n_threads=1,
        random_state=42,
    )
    synth.fit(
        real_df,
        exposure_col="exposure",
        frequency_col="claim_count",
        severity_col="claim_amount",
        categorical_cols=["region"],
        discrete_cols=["driver_age", "vehicle_group", "ncd_years", "annual_mileage", "claim_count"],
    )
    t_fit = time.time() - t0
    print(f"  Fit complete in {t_fit:.1f}s")

    print("Generating 5,000 synthetic policies...")
    t0 = time.time()
    vine_df = synth.generate(5_000)
    t_gen = time.time() - t0
    print(f"  Generated in {t_gen:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Metric 1: Frobenius norm of correlation matrix difference
    # -----------------------------------------------------------------------
    print("Metric 1: Spearman correlation preservation")
    print("-" * 50)
    frob_vine = spearman_frobenius(real_df, vine_df)
    frob_indep = spearman_frobenius(real_df, indep_df)
    print(f"  Frobenius norm (vine):        {frob_vine:.4f}  (lower = better)")
    print(f"  Frobenius norm (independent): {frob_indep:.4f}")
    print(f"  Vine is {frob_indep / max(frob_vine, 1e-9):.1f}x better at preserving correlations")
    print()

    # -----------------------------------------------------------------------
    # Metric 2: Marginal KS p-values
    # -----------------------------------------------------------------------
    print("Metric 2: Marginal distribution fidelity (KS p-values)")
    print("  p > 0.05 means the marginal is statistically indistinguishable from real")
    print("-" * 60)

    numeric_cols = [c for c in real_df.columns if real_df[c].dtype not in (pl.Utf8, pl.Categorical, pl.String)]

    pvals_vine = marginal_ks_pvalues(real_df, vine_df)
    pvals_indep = marginal_ks_pvalues(real_df, indep_df)

    col_w = [20, 16, 20]
    fmt = f"  {{:<{col_w[0]}}}  {{:<{col_w[1]}}}  {{:<{col_w[2]}}}"
    print(fmt.format("Column", "KS p-val (vine)", "KS p-val (indep)"))
    print("  " + "-" * 58)
    for col in numeric_cols:
        pv = pvals_vine.get(col, float("nan"))
        pi = pvals_indep.get(col, float("nan"))
        print(fmt.format(col, f"{pv:.4f}", f"{pi:.4f}"))
    print()

    # -----------------------------------------------------------------------
    # Metric 3: Pairwise mutual information for key pairs
    # -----------------------------------------------------------------------
    print("Metric 3: Pairwise mutual information (selected pairs)")
    print("  Higher MI = more dependency preserved")
    print("-" * 65)

    key_pairs = [
        ("ncd_years", "driver_age"),
        ("vehicle_group", "claim_count"),
        ("vehicle_group", "claim_amount"),
        ("ncd_years", "claim_count"),
    ]

    fmt3 = f"  {{:<35}}  {{:<10}}  {{:<10}}  {{:<10}}"
    print(fmt3.format("Pair", "Real", "Vine", "Independent"))
    print("  " + "-" * 63)

    for col_a, col_b in key_pairs:
        if col_a not in real_df.columns or col_b not in real_df.columns:
            continue
        mi_real = mutual_info_approx(
            real_df[col_a].to_numpy().astype(float),
            real_df[col_b].to_numpy().astype(float),
        )
        # Vine df may have different col names — use what's available
        if col_a in vine_df.columns and col_b in vine_df.columns:
            mi_vine = mutual_info_approx(
                vine_df[col_a].to_numpy().astype(float),
                vine_df[col_b].to_numpy().astype(float),
            )
        else:
            mi_vine = float("nan")
        mi_indep = mutual_info_approx(
            indep_df[col_a].to_numpy().astype(float),
            indep_df[col_b].to_numpy().astype(float),
        )
        print(fmt3.format(f"{col_a} vs {col_b}", f"{mi_real:.4f}", f"{mi_vine:.4f}", f"{mi_indep:.4f}"))
    print()

    # -----------------------------------------------------------------------
    # Metric 4: TSTR Gini gap
    # -----------------------------------------------------------------------
    print("Metric 4: TSTR — Train-on-Synthetic, Test-on-Real Gini gap")
    print("  Gini gap = real_gini - synth_gini. Near 0 = synthetic is useful for training")
    print("-" * 60)

    # Only use columns that exist in both synths
    common_cols = [c for c in real_df.columns if c in vine_df.columns and c in indep_df.columns]
    feature_cols = [c for c in common_cols if c != "claim_count"]

    print("  Testing vine synthesiser...")
    tstr_vine = tstr_gini_gap(
        real_df.select(common_cols),
        vine_df.select(common_cols),
        target_col="claim_count",
        feature_cols=feature_cols,
        seed=42,
    )

    print("  Testing independent shuffle...")
    tstr_indep = tstr_gini_gap(
        real_df.select(common_cols),
        indep_df.select(common_cols),
        target_col="claim_count",
        feature_cols=feature_cols,
        seed=42,
    )

    fmt4 = f"  {{:<20}}  {{:<12}}  {{:<14}}  {{:<10}}"
    print()
    print(fmt4.format("Method", "Real Gini", "Synth Gini", "Gap"))
    print("  " + "-" * 56)
    print(fmt4.format("Vine", f"{tstr_vine['real_gini']:.4f}", f"{tstr_vine['synth_gini']:.4f}", f"{tstr_vine['gap']:.4f}"))
    print(fmt4.format("Independent", f"{tstr_indep['real_gini']:.4f}", f"{tstr_indep['synth_gini']:.4f}", f"{tstr_indep['gap']:.4f}"))
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("Summary:")
    print(f"  Frobenius norm — vine: {frob_vine:.3f}  independent: {frob_indep:.3f}")
    print(f"  TSTR Gini gap  — vine: {tstr_vine['gap']:.4f}  independent: {tstr_indep['gap']:.4f}")
    print()
    print("  The vine copula preserves multivariate structure (low Frobenius")
    print("  norm, small TSTR gap). Independent shuffling destroys correlations")
    print("  while keeping marginals intact — models trained on it fail to")
    print("  generalise to real data.")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
