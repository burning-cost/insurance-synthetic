"""
Marginal distribution fitting for insurance columns.

Each column in a portfolio has its own distributional shape. Claim counts
are discrete (Poisson, NegBin). Premiums and severities are continuous and
right-skewed (Gamma, LogNormal). Driver age looks roughly Normal. NCD years
are bounded integers.

This module fits the best marginal for each column via AIC selection, then
wraps the result in a FittedMarginal that exposes .cdf(), .ppf(), and .rvs()
with a consistent interface — regardless of whether the underlying distribution
is scipy continuous, discrete, or a categorical lookup.

Design note: we avoid scipy's fitter() because it doesn't handle discrete
distributions or the exposure-offset needed for frequency columns. Rolling our
own gives us full control and keeps the API clean.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import polars as pl
from scipy import stats


# ---------------------------------------------------------------------------
# Candidate families
# ---------------------------------------------------------------------------

_CONTINUOUS_FAMILIES = [
    stats.gamma,
    stats.lognorm,
    stats.norm,
    stats.beta,
    stats.expon,
    stats.weibull_min,
]

_DISCRETE_FAMILIES = [
    stats.poisson,
    stats.nbinom,
]


# ---------------------------------------------------------------------------
# AIC helper
# ---------------------------------------------------------------------------

def _aic(log_likelihood: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood


def _fit_continuous(data: np.ndarray) -> tuple[stats.rv_continuous, tuple, float]:
    """Try each continuous family; return (dist, params, aic) for the best."""
    best_dist = None
    best_params: tuple = ()
    best_aic = np.inf

    data = data[np.isfinite(data)]
    if len(data) < 4:
        # Not enough data — fall back to Normal
        mu, sigma = float(np.mean(data)), max(float(np.std(data)), 1e-8)
        return stats.norm, (mu, sigma), _aic(
            np.sum(stats.norm.logpdf(data, mu, sigma)), 2
        )

    for dist in _CONTINUOUS_FAMILIES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Beta requires data in (0,1) — skip if not applicable
                if dist == stats.beta and (data.min() <= 0 or data.max() >= 1):
                    continue
                params = dist.fit(data, method="MLE")
                ll = float(np.sum(dist.logpdf(data, *params)))
                if not np.isfinite(ll):
                    continue
                aic = _aic(ll, len(params))
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist
                    best_params = params
        except Exception:
            continue

    if best_dist is None:
        mu, sigma = float(np.mean(data)), max(float(np.std(data)), 1e-8)
        best_dist = stats.norm
        best_params = (mu, sigma)
        best_aic = _aic(float(np.sum(stats.norm.logpdf(data, mu, sigma))), 2)

    return best_dist, best_params, best_aic


def _fit_discrete(data: np.ndarray) -> tuple[stats.rv_discrete, tuple, float]:
    """Try Poisson and NegBin; return best by AIC."""
    data = data[np.isfinite(data)].astype(int)
    data = data[data >= 0]

    best_dist = None
    best_params: tuple = ()
    best_aic = np.inf

    # Poisson: single parameter mu = mean
    mu = float(np.mean(data))
    if mu > 0:
        try:
            ll = float(np.sum(stats.poisson.logpmf(data, mu)))
            if np.isfinite(ll):
                aic = _aic(ll, 1)
                if aic < best_aic:
                    best_aic = aic
                    best_dist = stats.poisson
                    best_params = (mu,)
        except Exception:
            pass

    # NegBin: estimate n, p from method-of-moments
    mean_ = float(np.mean(data))
    var_ = float(np.var(data))
    if var_ > mean_ > 0:
        try:
            # n = mean^2 / (var - mean), p = mean / var
            n = mean_ ** 2 / (var_ - mean_)
            p = mean_ / var_
            n = max(n, 0.1)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = float(np.sum(stats.nbinom.logpmf(data, n, p)))
            if np.isfinite(ll):
                aic = _aic(ll, 2)
                if aic < best_aic:
                    best_aic = aic
                    best_dist = stats.nbinom
                    best_params = (n, p)
        except Exception:
            pass

    if best_dist is None:
        mu = max(float(np.mean(data)), 1e-6)
        best_dist = stats.poisson
        best_params = (mu,)
        best_aic = np.inf

    return best_dist, best_params, best_aic


# ---------------------------------------------------------------------------
# FittedMarginal
# ---------------------------------------------------------------------------

@dataclass
class FittedMarginal:
    """
    Fitted marginal distribution for a single column.

    Attributes
    ----------
    col_name : str
        Column name this marginal was fitted to.
    kind : str
        One of 'continuous', 'discrete', 'categorical'.
    dist : optional
        Fitted scipy distribution (None for categoricals).
    params : tuple
        Parameters passed to dist methods.
    aic : float
        AIC of the fitted distribution.
    categories : list
        For categorical columns: ordered list of category values. The integer
        encoding is the index into this list.
    cat_probs : np.ndarray
        For categorical columns: empirical probabilities of each category.
    clip_lower : float
        Hard lower bound applied during PPF (avoids degenerate samples).
    clip_upper : float
        Hard upper bound applied during PPF.
    """
    col_name: str
    kind: str  # 'continuous' | 'discrete' | 'categorical'
    dist: object = None
    params: tuple = field(default_factory=tuple)
    aic: float = np.inf
    categories: list = field(default_factory=list)
    cat_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    clip_lower: float = -np.inf
    clip_upper: float = np.inf

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function evaluated at x."""
        if self.kind == "categorical":
            # x is integer-encoded; CDF is cumulative over ordered categories
            x = np.asarray(x, dtype=int)
            cum = np.concatenate([[0.0], np.cumsum(self.cat_probs)])
            # CDF at integer k = P(X <= k)
            k = np.clip(x, 0, len(self.categories) - 1)
            return cum[k + 1]
        x = np.asarray(x, dtype=float)
        if self.kind == "discrete":
            return self.dist.cdf(x, *self.params)
        return self.dist.cdf(x, *self.params)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """
        Percent-point function (quantile function / inverse CDF).

        u should be in (0, 1). Returns values in the original scale.
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        if self.kind == "categorical":
            cum = np.concatenate([[0.0], np.cumsum(self.cat_probs)])
            # find index where cum[k] <= u < cum[k+1]
            indices = np.searchsorted(cum, u, side="right") - 1
            indices = np.clip(indices, 0, len(self.categories) - 1)
            return indices.astype(float)

        result = self.dist.ppf(u, *self.params)
        result = np.clip(result, self.clip_lower, self.clip_upper)
        return result

    def rvs(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw random samples."""
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0, 1, size)
        return self.ppf(u)

    def family_name(self) -> str:
        """Human-readable name of the fitted distribution."""
        if self.kind == "categorical":
            return f"Categorical({len(self.categories)} levels)"
        if self.dist is None:
            return "Unknown"
        return getattr(self.dist, "name", str(self.dist))


# ---------------------------------------------------------------------------
# Public API: fit_marginal
# ---------------------------------------------------------------------------

def fit_marginal(
    series: pl.Series,
    family: str = "auto",
    is_categorical: bool = False,
    is_discrete: bool = False,
) -> FittedMarginal:
    """
    Fit a marginal distribution to a Polars Series.

    Parameters
    ----------
    series : pl.Series
        The data column to fit. Nulls are dropped before fitting.
    family : str
        'auto' selects best family by AIC. Otherwise pass a scipy distribution
        name: 'gamma', 'lognorm', 'norm', 'poisson', 'nbinom'.
    is_categorical : bool
        Treat the column as categorical regardless of dtype. Encodes as
        integers 0..n_categories-1 and stores the category mapping.
    is_discrete : bool
        Force discrete distribution families (Poisson, NegBin). If False and
        dtype is integer, we still try continuous families unless `is_discrete`
        is explicitly set.

    Returns
    -------
    FittedMarginal
        Fitted marginal with .cdf(), .ppf(), .rvs() methods.
    """
    col_name = series.name
    data = series.drop_nulls()

    # --- Categorical handling ---
    if is_categorical or data.dtype == pl.Utf8 or data.dtype == pl.Categorical:
        counts = data.value_counts(sort=True)
        categories = counts[""].to_list() if "" in counts.columns else counts[counts.columns[0]].to_list()
        # polars value_counts column name is the series name
        cat_col = col_name if col_name in counts.columns else counts.columns[0]
        categories = counts[cat_col].to_list()
        total = len(data)
        probs = np.array([c / total for c in counts["count"].to_list()], dtype=float)
        return FittedMarginal(
            col_name=col_name,
            kind="categorical",
            categories=categories,
            cat_probs=probs,
        )

    arr = data.to_numpy().astype(float)

    if len(arr) == 0:
        raise ValueError(f"Column '{col_name}' has no non-null values after dropping nulls.")

    # --- Discrete handling ---
    if is_discrete or (data.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
                       and arr.min() >= 0 and family == "auto"):
        if family != "auto":
            family_map = {"poisson": stats.poisson, "nbinom": stats.nbinom}
            dist = family_map.get(family)
            if dist is not None:
                if dist == stats.poisson:
                    mu = float(np.mean(arr))
                    params: tuple = (max(mu, 1e-6),)
                else:
                    mean_ = float(np.mean(arr))
                    var_ = float(np.var(arr))
                    n = mean_ ** 2 / max(var_ - mean_, 1e-6)
                    p = mean_ / max(var_, 1e-6)
                    params = (max(n, 0.1), np.clip(p, 1e-6, 1 - 1e-6))
                ll = float(np.sum(dist.logpmf(arr.astype(int), *params)))
                return FittedMarginal(
                    col_name=col_name,
                    kind="discrete",
                    dist=dist,
                    params=params,
                    aic=_aic(ll, len(params)),
                    clip_lower=0.0,
                )

        dist, params, aic = _fit_discrete(arr)
        return FittedMarginal(
            col_name=col_name,
            kind="discrete",
            dist=dist,
            params=params,
            aic=aic,
            clip_lower=0.0,
        )

    # --- Continuous handling ---
    clip_lower = float(arr.min()) if arr.min() >= 0 else -np.inf
    clip_upper = np.inf

    if family != "auto":
        family_map = {
            "gamma": stats.gamma, "lognorm": stats.lognorm, "norm": stats.norm,
            "beta": stats.beta, "expon": stats.expon, "weibull": stats.weibull_min,
        }
        dist_cls = family_map.get(family)
        if dist_cls is None:
            raise ValueError(f"Unknown family '{family}'. Use 'auto' or one of: {list(family_map)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dist_cls.fit(arr, method="MLE")
        ll = float(np.sum(dist_cls.logpdf(arr, *params)))
        return FittedMarginal(
            col_name=col_name,
            kind="continuous",
            dist=dist_cls,
            params=params,
            aic=_aic(ll, len(params)),
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )

    dist, params, aic = _fit_continuous(arr)
    return FittedMarginal(
        col_name=col_name,
        kind="continuous",
        dist=dist,
        params=params,
        aic=aic,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
