"""
InsuranceSynthesizer: the main entry point for synthetic portfolio generation.

The workflow is:
    1. fit() — fit a marginal distribution per column, then a vine copula
               on the probability-integral-transformed (PIT) data.
    2. generate() — sample from the vine copula, invert through marginals.
    3. summary() — print fitted marginals and copula structure.

Exposure handling deserves a note. The exposure column is not modelled
independently — it is fitted as a marginal and reproduced by the copula
like any other column. But the frequency column is *conditional* on exposure
and risk factors: we draw from Poisson(lambda_i * exposure_i) where lambda_i
is the empirical frequency rate for the risk group that observation belongs to.

This is important. A flat portfolio-average rate ignores all risk factor
variation and produces synthetic claim counts that are uncorrelated with
the factors that actually drive frequency. We compute empirical rates per
unique combination of categorical risk factors from the training data, then
look up the appropriate rate for each synthetic row when generating. Rows
whose risk-factor combination does not appear in the training data (very rare
combinations) fall back to the overall portfolio average rate.

Severity handling: the claim_amount column is excluded from the vine copula.
Zero-inflated columns (most policies have no claims) break continuous copula
fitting — the massive point mass at zero causes the fitted marginal CDF to
collapse, producing KS statistics near 1.0. Instead, we:

    1. Fit the severity marginal on non-zero observations only (the actual
       severity distribution, conditional on a claim occurring).
    2. In generate(), draw severity independently from the fitted marginal
       for each row where claim_count > 0. Non-claimers get severity = 0.

This is the actuarially correct treatment: severity is a separate model from
frequency. Joint modelling via the copula would require a copula that handles
zero-inflation, which pyvinecopulib does not expose cleanly.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import polars as pl

from ._marginals import FittedMarginal, fit_marginal
from ._copula import VineCopulaModel


_DEFAULT_SEED = 42


class InsuranceSynthesizer:
    """
    Generate synthetic insurance portfolio data using vine copulas.

    Parameters
    ----------
    method : str
        Dependence model. 'vine' uses pyvinecopulib R-vines (recommended).
        'gaussian' forces a Gaussian copula (faster, no tail dependence).
    marginals : str or dict
        'auto' selects the best marginal for each column by AIC.
        Pass a dict mapping column name -> scipy distribution name to
        override specific columns (e.g. {'driver_age': 'norm'}).
    family_set : str
        Vine copula family set. 'all' (default) lets pyvinecopulib choose
        the best bivariate copula for each pair.
    trunc_lvl : int, optional
        Vine truncation level. None = full vine. For high-dimensional data
        (many rating factors), truncating at 3 gives a good speed/quality
        tradeoff.
    n_threads : int
        Threads for vine copula fitting. Default 1.
    random_state : int or np.random.Generator, optional
        Seed for reproducibility.

    Examples
    --------
    >>> import polars as pl
    >>> from insurance_synthetic import InsuranceSynthesizer
    >>>
    >>> synth = InsuranceSynthesizer()
    >>> synth.fit(df, exposure_col='exposure', frequency_col='claim_count')
    >>> synthetic_df = synth.generate(10_000)
    >>> print(synth.summary())
    """

    def __init__(
        self,
        method: str = "vine",
        marginals: Union[str, dict] = "auto",
        family_set: str = "all",
        trunc_lvl: Optional[int] = None,
        n_threads: int = 1,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        if method not in ("vine", "gaussian"):
            raise ValueError(f"method must be 'vine' or 'gaussian', got '{method}'")
        self.method = method
        self.marginals_spec = marginals
        self.family_set = "gaussian" if method == "gaussian" else family_set
        self.trunc_lvl = trunc_lvl
        self.n_threads = n_threads

        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        elif random_state is not None:
            self._rng = np.random.default_rng(int(random_state))
        else:
            self._rng = np.random.default_rng(_DEFAULT_SEED)

        # Set after fitting
        self._fitted_marginals: dict[str, FittedMarginal] = {}
        self._severity_marginal: Optional[FittedMarginal] = None  # fitted on non-zero only
        self._copula: Optional[VineCopulaModel] = None
        self._columns: list[str] = []  # columns passed to the vine copula (excludes severity)
        self._all_columns: list[str] = []  # original column order (for output)
        self._exposure_col: Optional[str] = None
        self._frequency_col: Optional[str] = None
        self._severity_col: Optional[str] = None
        self._frequency_rate: Optional[float] = None  # portfolio-average fallback rate
        self._frequency_rate_table: Optional[dict] = None  # per risk-group rates
        self._frequency_rate_cols: Optional[list[str]] = None  # cols used for grouping
        self._categorical_cols: set[str] = set()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        exposure_col: str = "exposure",
        frequency_col: str = "claim_count",
        severity_col: Optional[str] = None,
        categorical_cols: Optional[list[str]] = None,
        discrete_cols: Optional[list[str]] = None,
    ) -> "InsuranceSynthesizer":
        """
        Fit marginal distributions and a vine copula to the portfolio data.

        Parameters
        ----------
        df : pl.DataFrame
            The portfolio data. All numeric and string columns are used.
        exposure_col : str
            Name of the exposure column (in policy years). This column is
            fitted as a marginal and reproduced by the copula. If not present
            in df, exposure is assumed to be 1.0 for all rows.
        frequency_col : str
            Name of the claim count column. If present, its marginal is fitted
            but generation uses Poisson(lambda_i * exposure_i) to preserve the
            exposure/frequency relationship, where lambda_i is the empirical
            rate for the risk-factor group of observation i.
        severity_col : str, optional
            Name of the claim amount column. Excluded from the vine copula —
            instead, a severity marginal is fitted on non-zero observations
            only (the conditional severity distribution). Generation draws
            from this marginal for rows where claim_count > 0; all other rows
            get severity = 0. This is the correct actuarial treatment of
            zero-inflated severity.
        categorical_cols : list of str, optional
            Columns to treat as categorical. Auto-detected for string/Utf8
            columns; pass this to force integer columns to be treated as
            categorical (e.g. region_code).
        discrete_cols : list of str, optional
            Columns to fit with discrete distributions (Poisson / NegBin).
            Integer columns are auto-detected as discrete.

        Returns
        -------
        self
        """
        if exposure_col not in df.columns:
            warnings.warn(
                f"exposure_col '{exposure_col}' not found in DataFrame. "
                "Assuming exposure = 1.0 for all rows.",
                UserWarning,
                stacklevel=2,
            )
            df = df.with_columns(pl.lit(1.0).alias(exposure_col))

        self._exposure_col = exposure_col
        self._frequency_col = frequency_col if frequency_col in df.columns else None
        self._severity_col = severity_col if severity_col and severity_col in df.columns else None
        self._all_columns = df.columns
        self._categorical_cols = set(categorical_cols or [])

        # Auto-detect string columns as categorical
        for col in df.columns:
            if df[col].dtype in (pl.Utf8, pl.Categorical, pl.String):
                self._categorical_cols.add(col)

        force_discrete = set(discrete_cols or [])

        # Fit marginal per column (including severity — used in summary())
        marginal_overrides: dict = {}
        if isinstance(self.marginals_spec, dict):
            marginal_overrides = self.marginals_spec

        for col in df.columns:
            family = marginal_overrides.get(col, "auto")
            is_cat = col in self._categorical_cols
            is_disc = col in force_discrete

            marginal = fit_marginal(
                df[col],
                family=family,
                is_categorical=is_cat,
                is_discrete=is_disc,
            )
            self._fitted_marginals[col] = marginal

        # Fit a separate severity marginal on non-zero claim amounts only.
        # Zero-inflated columns break continuous copula fitting: the huge
        # point mass at zero collapses the CDF and produces KS near 1.0.
        # We exclude severity from the vine and handle it separately at
        # generation time (severity given a claim occurred).
        if self._severity_col is not None:
            sev_series = df[self._severity_col]
            nonzero_sev = sev_series.filter(sev_series > 0)
            if len(nonzero_sev) >= 4:
                sev_family = marginal_overrides.get(self._severity_col, "auto")
                self._severity_marginal = fit_marginal(
                    nonzero_sev,
                    family=sev_family,
                    is_categorical=False,
                    is_discrete=False,
                )
            else:
                warnings.warn(
                    f"severity_col '{self._severity_col}' has fewer than 4 non-zero "
                    "observations. Severity synthesis will produce zeros only.",
                    UserWarning,
                    stacklevel=2,
                )
                self._severity_marginal = None

        # Compute frequency rates: per risk-factor group and portfolio average
        if self._frequency_col is not None:
            total_claims = float(df[self._frequency_col].sum())
            total_exposure = float(df[self._exposure_col].sum())
            self._frequency_rate = total_claims / max(total_exposure, 1e-9)
            # Build per-group rate table using categorical columns (excluding
            # frequency and exposure themselves) as grouping keys.
            rate_cols = [
                c for c in df.columns
                if c in self._categorical_cols
                and c != self._frequency_col
                and c != self._exposure_col
            ]
            if rate_cols:
                rate_table = _compute_group_rates(df, rate_cols, frequency_col, exposure_col)
                self._frequency_rate_table = rate_table
                self._frequency_rate_cols = rate_cols
            else:
                # No categorical grouping columns — fall back to portfolio average
                self._frequency_rate_table = None
                self._frequency_rate_cols = None

        # Columns passed to the vine copula: exclude severity (handled separately)
        copula_cols = [c for c in df.columns if c != self._severity_col]
        self._columns = copula_cols

        # PIT transform copula columns to uniform [0,1]
        u_matrix = self._pit_transform(df.select(copula_cols))

        # Fit vine copula
        self._copula = VineCopulaModel(
            family_set=self.family_set,
            trunc_lvl=self.trunc_lvl,
            n_threads=self.n_threads,
        )
        self._copula.fit(u_matrix)

        self._is_fitted = True
        return self

    def _pit_transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        Apply the probability integral transform to each column.

        Returns an (n, d) array of uniform [0,1] values. Categoricals are
        mapped through their empirical CDF. Ties in discrete columns are
        resolved by adding uniform jitter in [prev_u, u] — the standard
        approach for fitting copulas to discrete data.

        For zero counts (the common case in motor — 85%+ of policies have
        zero claims), prev_u is set to 0.0 rather than CDF(0 - 1) = CDF(-1),
        which would be CDF(0) due to clipping and collapse to a point mass.
        """
        n = len(df)
        d = len(df.columns)
        u = np.empty((n, d), dtype=float)

        for j, col in enumerate(df.columns):
            marginal = self._fitted_marginals[col]

            if marginal.kind == "categorical":
                # Convert string/category values to integer indices
                cat_map = {v: i for i, v in enumerate(marginal.categories)}
                raw_vals = df[col].to_list()
                arr = np.array([cat_map.get(v, 0) for v in raw_vals], dtype=float)
            else:
                arr = df[col].to_numpy().astype(float)

            raw_u = marginal.cdf(arr)

            # Discrete / categorical jitter to avoid ties at CDF steps
            if marginal.kind in ("discrete", "categorical"):
                # Step back one CDF increment, then add uniform jitter across the step.
                # For arr == 0 (or any arr at the left boundary), prev_u = 0.0 rather
                # than CDF(arr - 1) — because for arr=0, CDF(0-1) clips to CDF(0) = raw_u
                # which makes jitter width zero, collapsing to a point mass. This is
                # especially important for motor claim counts where ~85% of rows are zero.
                prev_u = _discrete_prev_cdf(marginal, arr)
                raw_u = prev_u + (raw_u - prev_u) * self._rng.uniform(0, 1, size=n)

            u[:, j] = np.clip(raw_u, 1e-6, 1 - 1e-6)

        return u

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int,
        constraints: Optional[dict] = None,
        max_resample_attempts: int = 10,
    ) -> pl.DataFrame:
        """
        Generate n synthetic policies.

        Parameters
        ----------
        n : int
            Number of rows to generate.
        constraints : dict, optional
            Column-level constraints. Each value can be:
            - A tuple (min, max): values outside this range are resampled.
            - A callable: rows where callable(value) is False are resampled.
            Example: {'driver_age': (17, 90), 'ncd_years': (0, 25)}
        max_resample_attempts : int
            How many rounds of resample-and-replace to attempt for constraint
            violations. Violations remaining after this are clamped or left.

        Returns
        -------
        pl.DataFrame
            Synthetic portfolio with the same columns as the training data.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")
        if n <= 0:
            raise ValueError(
                f"n must be a positive integer, got {n}. "
                "generate() requires at least 1 sample."
            )

        # Generate slightly more than needed in case constraints remove some rows
        oversample = max(n, int(n * 1.2))
        rows = self._generate_raw(oversample)

        # Apply constraints
        if constraints:
            rows = self._apply_constraints(rows, constraints, n, max_resample_attempts)

        # Trim to exactly n rows
        rows = rows[:n]

        # Regenerate frequency column conditional on exposure and risk factors.
        # We use per-group empirical rates rather than a single portfolio average.
        # This ensures the synthetic claim counts are correlated with the risk
        # factors that drive frequency, not just with exposure.
        if self._frequency_col is not None and self._frequency_rate is not None:
            exposure = rows[self._exposure_col].to_numpy()
            lambdas = self._compute_row_rates(rows) * exposure
            counts = self._rng.poisson(lambdas)
            clip_upper = self._fitted_marginals[self._frequency_col].clip_upper
            if np.isfinite(clip_upper):
                counts = np.clip(counts, 0, int(clip_upper))
            else:
                counts = np.clip(counts, 0, None)

            rows = rows.with_columns(
                pl.Series(name=self._frequency_col, values=counts.astype(int))
            )
        else:
            counts = None

        # Regenerate severity column independently for each claimer row.
        # Severity is NOT modelled through the vine copula — zero-inflation
        # in the raw column makes copula fitting fail (KS near 1.0). Instead:
        #   - rows with claim_count > 0 get severity drawn from the conditional
        #     severity marginal (fitted on non-zero claims from training data)
        #   - rows with claim_count == 0 get severity = 0.0
        if self._severity_col is not None:
            if counts is not None:
                has_claim = counts > 0
            elif self._frequency_col is not None and self._frequency_col in rows.columns:
                has_claim = rows[self._frequency_col].to_numpy() > 0
            else:
                # No frequency information — draw severity for all rows
                has_claim = np.ones(n, dtype=bool)

            severity = np.zeros(n, dtype=float)
            n_claimers = int(has_claim.sum())

            if n_claimers > 0 and self._severity_marginal is not None:
                u_sev = self._rng.uniform(0, 1, size=n_claimers)
                sev_values = self._severity_marginal.ppf(u_sev)
                sev_values = np.maximum(sev_values, 0.0)
                severity[has_claim] = sev_values

            rows = rows.with_columns(
                pl.Series(name=self._severity_col, values=severity.tolist(), dtype=pl.Float64)
            )

        # Reorder columns to match original input order (severity was excluded from vine)
        output_cols = [c for c in self._all_columns if c in rows.columns]
        rows = rows.select(output_cols)

        return rows

    def _compute_row_rates(self, rows: pl.DataFrame) -> np.ndarray:
        """
        Look up the empirical frequency rate for each row.

        For rows whose risk-factor combination appears in the training data,
        return the empirical rate for that group. For all other rows, fall back
        to the portfolio average rate.

        Returns an array of per-row annualised frequency rates (claims/year).
        """
        n = len(rows)
        rates = np.full(n, self._frequency_rate, dtype=float)

        if self._frequency_rate_table is None or not self._frequency_rate_cols:
            return rates

        rate_cols = self._frequency_rate_cols
        rate_table = self._frequency_rate_table

        # Build a tuple key per row from the grouping columns and look up rates
        col_arrays = [rows[c].to_list() for c in rate_cols]
        for i in range(n):
            key = tuple(col_arrays[j][i] for j in range(len(rate_cols)))
            if key in rate_table:
                rates[i] = rate_table[key]

        return rates

    def _generate_raw(self, n: int) -> pl.DataFrame:
        """
        Sample from the vine copula and invert through marginals.
        Returns a DataFrame in the original column scale.
        Note: severity column is excluded — it is handled in generate().
        """
        u = self._copula.simulate(n, rng=self._rng)  # (n, d)

        data: dict[str, list] = {}
        for j, col in enumerate(self._columns):
            marginal = self._fitted_marginals[col]
            col_u = u[:, j]
            values = marginal.ppf(col_u)

            if marginal.kind == "categorical":
                # Map integer indices back to category labels
                indices = values.astype(int)
                indices = np.clip(indices, 0, len(marginal.categories) - 1)
                data[col] = [marginal.categories[i] for i in indices]
            elif marginal.kind == "discrete":
                data[col] = values.astype(int).tolist()
            else:
                data[col] = values.tolist()

        # Re-type columns to match the original dtypes
        series_list = []
        for col, vals in data.items():
            marginal = self._fitted_marginals[col]
            if marginal.kind == "categorical":
                series_list.append(pl.Series(name=col, values=vals, dtype=pl.Utf8))
            elif marginal.kind == "discrete":
                series_list.append(pl.Series(name=col, values=vals, dtype=pl.Int64))
            else:
                series_list.append(pl.Series(name=col, values=vals, dtype=pl.Float64))

        return pl.DataFrame(series_list)

    def _apply_constraints(
        self,
        df: pl.DataFrame,
        constraints: dict,
        target_n: int,
        max_attempts: int,
    ) -> pl.DataFrame:
        """
        Iteratively resample rows that violate constraints.

        We keep a running pool of valid rows and top up from fresh vine
        samples until we have enough, or exhaust max_attempts.
        """
        valid = df
        attempt = 0

        while attempt < max_attempts:
            mask = _build_valid_mask(valid, constraints)
            valid = valid.filter(mask)
            if len(valid) >= target_n:
                break
            needed = max(target_n - len(valid), int(target_n * 0.2))
            fresh = self._generate_raw(needed * 2)
            valid = pl.concat([valid, fresh])
            attempt += 1

        if len(valid) < target_n:
            warnings.warn(
                f"After {max_attempts} resample attempts, only {len(valid)} valid rows "
                f"(target: {target_n}). Some constraint violations remain; "
                "consider loosening constraints or increasing max_resample_attempts.",
                UserWarning,
                stacklevel=3,
            )

        return valid

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Print a summary of the fitted marginals and copula.

        Returns the summary as a string (also printed to stdout).
        """
        if not self._is_fitted:
            return "InsuranceSynthesizer (not fitted)"

        lines = ["InsuranceSynthesizer — fitted model summary", "=" * 50]

        lines.append(f"\nMethod: {self.method}")
        if self._frequency_col:
            lines.append(
                f"Frequency column: '{self._frequency_col}' "
                f"(portfolio average rate = {self._frequency_rate:.4f} claims/year)"
            )
            if self._frequency_rate_cols:
                n_groups = len(self._frequency_rate_table) if self._frequency_rate_table else 0
                lines.append(
                    f"  Per-group rates computed on: {self._frequency_rate_cols} "
                    f"({n_groups} unique groups)"
                )
        if self._exposure_col:
            lines.append(f"Exposure column: '{self._exposure_col}'")
        if self._severity_col:
            lines.append(
                f"Severity column: '{self._severity_col}' "
                f"(fitted independently on non-zero claims, excluded from vine copula)"
            )

        lines.append("\nFitted marginals:")
        lines.append(f"  {'Column':<25} {'Kind':<12} {'Family':<25} {'AIC':>10}  {'Note'}")
        lines.append("  " + "-" * 85)
        for col, m in self._fitted_marginals.items():
            aic_str = f"{m.aic:.1f}" if np.isfinite(m.aic) else "—"
            note = ""
            if col == self._severity_col:
                if self._severity_marginal is not None:
                    note = f"(copula-excluded; conditional marginal: {self._severity_marginal.family_name()})"
                else:
                    note = "(copula-excluded; insufficient non-zero claims)"
            lines.append(
                f"  {col:<25} {m.kind:<12} {m.family_name():<25} {aic_str:>10}  {note}"
            )

        lines.append("\nCopula:")
        if self._copula is not None:
            lines.append("  " + self._copula.summary().replace("\n", "\n  "))

        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    # Serialisation (basic)
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return constructor parameters."""
        return {
            "method": self.method,
            "marginals": self.marginals_spec,
            "family_set": self.family_set,
            "trunc_lvl": self.trunc_lvl,
            "n_threads": self.n_threads,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_group_rates(
    df: pl.DataFrame,
    rate_cols: list[str],
    frequency_col: str,
    exposure_col: str,
) -> dict:
    """
    Compute empirical frequency rates per unique combination of rate_cols.

    Returns a dict mapping tuple(group values) -> rate (claims / exposure).
    Groups with zero exposure are excluded.
    """
    # Group by all categorical columns and sum claims and exposure
    agg = df.group_by(rate_cols).agg([
        pl.col(frequency_col).sum().alias("_claims"),
        pl.col(exposure_col).sum().alias("_exposure"),
    ])

    rate_table = {}
    for row in agg.iter_rows(named=True):
        exposure = row["_exposure"]
        if exposure > 0:
            key = tuple(row[c] for c in rate_cols)
            rate_table[key] = row["_claims"] / exposure

    return rate_table


def _discrete_prev_cdf(marginal: FittedMarginal, arr: np.ndarray) -> np.ndarray:
    """
    Return CDF at arr - 1 for discrete marginals, clipped to [0, 1].

    This is the left limit of the CDF just before the jump at each value,
    used to uniformly jitter discrete draws across the CDF step.

    For arr <= 0 (the common zero-count case), we return 0.0 directly.
    Without this fix, arr=0 would compute CDF(clip(0-1, 0, None)) = CDF(0),
    which equals CDF(arr), giving zero jitter width and collapsing 85%+ of
    motor policies to a point mass at zero in the copula.
    """
    prev_u = np.where(arr <= 0, 0.0, marginal.cdf(arr - 1))
    return np.clip(prev_u, 0.0, 1.0)


def _build_valid_mask(df: pl.DataFrame, constraints: dict) -> pl.Series:
    """
    Build a boolean Series that is True for rows satisfying all constraints.
    """
    mask = pl.Series(values=[True] * len(df))

    for col, rule in constraints.items():
        if col not in df.columns:
            continue
        col_vals = df[col]

        if isinstance(rule, tuple) and len(rule) == 2:
            lo, hi = rule
            col_mask = (col_vals >= lo) & (col_vals <= hi)
        elif callable(rule):
            col_mask = pl.Series(values=[bool(rule(v)) for v in col_vals.to_list()])
        else:
            raise ValueError(
                f"Constraint for '{col}' must be a (min, max) tuple or a callable. "
                f"Got: {type(rule)}"
            )

        mask = mask & col_mask

    return mask
