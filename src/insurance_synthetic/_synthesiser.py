"""
InsuranceSynthesizer: the main entry point for synthetic portfolio generation.

The workflow is:
    1. fit() — fit a marginal distribution per column, then a vine copula
               on the probability-integral-transformed (PIT) data.
    2. generate() — sample from the vine copula, invert through marginals,
                    regenerate frequency columns using exposure offset.
    3. summary() — print fitted marginals and copula structure.

Exposure handling deserves a note. The exposure column is not modelled
independently — it is fitted as a marginal and reproduced by the copula
like any other column. But the frequency column is *conditional* on exposure:
we draw from Poisson(lambda * exposure) where lambda is the fitted rate,
not simply invert the frequency marginal. This preserves the frequency/
exposure relationship that actuaries depend on (e.g., annualised claim rate
should be stable regardless of how we slice the portfolio).
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
        self._copula: Optional[VineCopulaModel] = None
        self._columns: list[str] = []
        self._exposure_col: Optional[str] = None
        self._frequency_col: Optional[str] = None
        self._severity_col: Optional[str] = None
        self._frequency_rate: Optional[float] = None  # lambda = claims / exposure
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
            but generation uses Poisson(lambda * exposure) to preserve the
            exposure/frequency relationship.
        severity_col : str, optional
            Name of the claim amount column. Fitted as a continuous marginal.
            Zero-inflated (most policies have no claims) — the zero mass is
            absorbed by placing a point mass at the lower tail.
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
        self._columns = df.columns
        self._categorical_cols = set(categorical_cols or [])

        # Auto-detect string columns as categorical
        for col in df.columns:
            if df[col].dtype in (pl.Utf8, pl.Categorical, pl.String):
                self._categorical_cols.add(col)

        force_discrete = set(discrete_cols or [])

        # Fit marginal per column
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

        # Compute frequency rate: overall lambda = total_claims / total_exposure
        if self._frequency_col is not None:
            total_claims = float(df[self._frequency_col].sum())
            total_exposure = float(df[self._exposure_col].sum())
            self._frequency_rate = total_claims / max(total_exposure, 1e-9)

        # PIT transform all columns to uniform [0,1]
        u_matrix = self._pit_transform(df)

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
        resolved by adding uniform jitter in [0, 1/n] — the standard approach
        for fitting copulas to discrete data.
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
                # Step back one CDF increment, then add uniform jitter across the step
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

        # Generate slightly more than needed in case constraints remove some rows
        oversample = max(n, int(n * 1.2))
        rows = self._generate_raw(oversample)

        # Apply constraints
        if constraints:
            rows = self._apply_constraints(rows, constraints, n, max_resample_attempts)

        # Trim to exactly n rows
        rows = rows[:n]

        # Regenerate frequency column using exposure offset
        if self._frequency_col is not None and self._frequency_rate is not None:
            exposure = rows[self._exposure_col].to_numpy()
            lambdas = self._frequency_rate * exposure
            counts = self._rng.poisson(lambdas)
            clip_upper = self._fitted_marginals[self._frequency_col].clip_upper
            if np.isfinite(clip_upper):
                counts = np.clip(counts, 0, int(clip_upper))
            rows = rows.with_columns(
                pl.Series(name=self._frequency_col, values=counts.astype(int))
            )

        return rows

    def _generate_raw(self, n: int) -> pl.DataFrame:
        """
        Sample from the vine copula and invert through marginals.
        Returns a DataFrame in the original column scale.
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
                f"(rate lambda = {self._frequency_rate:.4f} claims/year)"
            )
        if self._exposure_col:
            lines.append(f"Exposure column: '{self._exposure_col}'")

        lines.append("\nFitted marginals:")
        lines.append(f"  {'Column':<25} {'Kind':<12} {'Family':<25} {'AIC':>10}")
        lines.append("  " + "-" * 75)
        for col, m in self._fitted_marginals.items():
            aic_str = f"{m.aic:.1f}" if np.isfinite(m.aic) else "—"
            lines.append(
                f"  {col:<25} {m.kind:<12} {m.family_name():<25} {aic_str:>10}"
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

def _discrete_prev_cdf(marginal: FittedMarginal, arr: np.ndarray) -> np.ndarray:
    """
    Return CDF at arr - 1 for discrete marginals, clipped to [0, 1].

    This is the left limit of the CDF just before the jump at each value,
    used to uniformly jitter discrete draws across the CDF step.
    """
    prev = np.clip(arr - 1, 0, None)
    return np.clip(marginal.cdf(prev), 0.0, 1.0)


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
