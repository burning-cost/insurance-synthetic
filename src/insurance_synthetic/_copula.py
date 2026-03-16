"""
Vine copula wrapper for insurance portfolio synthesis.

We use pyvinecopulib's Vinecop class as the engine. It fits a regular vine
(R-vine) using maximum likelihood, with automatic family selection from the
full set including Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, and
their rotations. This matters for insurance data: claim frequency is right-
tailed, and the Gaussian copula would miss the lower tail dependence between
bad-risk indicators (high vehicle group + low NCD + young driver).

If pyvinecopulib is not importable (e.g. on an unusual platform), we fall back
to a Gaussian copula fitted via the correlation matrix of the PIT-transformed
data. The Gaussian fallback preserves linear correlations but misses tail
dependence. It is documented clearly so users know what they're getting.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Try to import pyvinecopulib; prepare fallback
# ---------------------------------------------------------------------------

try:
    import pyvinecopulib as pv  # type: ignore
    _VINE_AVAILABLE = True
except ImportError:
    _VINE_AVAILABLE = False
    pv = None  # type: ignore


class VineCopulaModel:
    """
    Thin wrapper around pyvinecopulib.Vinecop (or Gaussian fallback).

    After fitting, exposes .simulate() to draw uniform samples that
    match the dependence structure of the training data.

    Parameters
    ----------
    family_set : str
        Copula family set passed to pyvinecopulib. 'all' lets it choose the
        best bivariate copula for each pair. 'parametric' excludes empirical.
        'gaussian' forces Gaussian copulas throughout (faster but weaker).
    trunc_lvl : int, optional
        Truncation level for the vine (number of trees). None = full vine.
        For portfolios with many columns, truncating at 3-4 trees is often
        sufficient and much faster to fit.
    n_threads : int
        Number of threads for pyvinecopulib fitting. Default 1 to avoid
        resource contention on the Raspberry Pi.
    """

    def __init__(
        self,
        family_set: str = "all",
        trunc_lvl: Optional[int] = None,
        n_threads: int = 1,
    ):
        self.family_set = family_set
        self.trunc_lvl = trunc_lvl
        self.n_threads = n_threads
        self._vine: object = None
        self._corr_matrix: Optional[np.ndarray] = None  # for Gaussian fallback
        self._n_vars: int = 0
        self._fitted: bool = False
        self._using_fallback: bool = not _VINE_AVAILABLE

    def fit(self, u: np.ndarray) -> "VineCopulaModel":
        """
        Fit the vine copula to PIT-transformed data.

        Parameters
        ----------
        u : np.ndarray, shape (n, d)
            Data in uniform [0,1] margins (i.e. after applying the empirical
            or parametric CDF to each column). Each column must be in (0,1).

        Returns
        -------
        self
        """
        n, d = u.shape
        self._n_vars = d

        u = np.clip(u, 1e-6, 1 - 1e-6)

        if _VINE_AVAILABLE and not self._using_fallback:
            self._fit_vine(u)
        else:
            if self._using_fallback and not _VINE_AVAILABLE:
                # Vine library not installed — warn the user
                warnings.warn(
                    "pyvinecopulib is not installed. Falling back to Gaussian copula. "
                    "Install insurance-synthetic[vine] for full vine copula support.",
                    ImportWarning,
                    stacklevel=3,
                )
            self._fit_gaussian(u)

        self._fitted = True
        return self

    def _fit_vine(self, u: np.ndarray) -> None:
        controls = pv.FitControlsVinecop(
            family_set=_parse_family_set(self.family_set),
            trunc_lvl=self.trunc_lvl if self.trunc_lvl is not None else 1000,
            num_threads=self.n_threads,
        )
        # pyvinecopulib >=0.7 uses factory method Vinecop.from_data().
        # Older versions accepted Vinecop(data=u, controls=controls).
        if hasattr(pv.Vinecop, "from_data"):
            self._vine = pv.Vinecop.from_data(u, controls=controls)
        else:
            self._vine = pv.Vinecop(data=u, controls=controls)

    def _fit_gaussian(self, u: np.ndarray) -> None:
        """
        Gaussian copula fallback: estimate the correlation matrix from
        normal scores (rank-based correlation).
        """
        from scipy.stats import norm as sp_norm
        z = sp_norm.ppf(u)
        self._corr_matrix = np.corrcoef(z.T)
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(self._corr_matrix)
        if eigvals.min() < 1e-8:
            self._corr_matrix += np.eye(self._n_vars) * (1e-6 - eigvals.min())

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n samples from the fitted copula.

        Returns
        -------
        np.ndarray, shape (n, d)
            Samples in uniform [0,1] margins.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before simulate().")

        seed = int(rng.integers(0, 2**31)) if rng is not None else None

        if _VINE_AVAILABLE and not self._using_fallback:
            u = self._vine.simulate(n, seeds=[seed] if seed is not None else [])
        else:
            u = self._simulate_gaussian(n, rng)

        return u

    def _simulate_gaussian(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        from scipy.stats import norm as sp_norm
        if rng is None:
            rng = np.random.default_rng()
        L = np.linalg.cholesky(self._corr_matrix)
        z = rng.standard_normal((n, self._n_vars)) @ L.T
        return sp_norm.cdf(z)

    @property
    def using_fallback(self) -> bool:
        """True if using Gaussian fallback instead of vine copula."""
        return self._using_fallback

    def summary(self) -> str:
        """Human-readable summary of the fitted copula."""
        if not self._fitted:
            return "VineCopulaModel (not fitted)"
        if self._using_fallback:
            return (
                f"Gaussian copula fallback (pyvinecopulib unavailable)\n"
                f"  Variables: {self._n_vars}\n"
                f"  Correlation matrix shape: {self._corr_matrix.shape}"
            )
        return (
            f"Vine copula (pyvinecopulib)\n"
            f"  Variables: {self._n_vars}\n"
            f"  Family set: {self.family_set}\n"
            f"  Truncation level: {self.trunc_lvl}\n"
            f"  Trees: {self._vine}"
        )


def _parse_family_set(name: str):
    """Convert string name to pyvinecopulib family set."""
    if not _VINE_AVAILABLE:
        return None
    mapping = {
        "all": pv.all,
        "parametric": pv.parametric,
        "nonparametric": pv.nonparametric,
        "gaussian": [pv.BicopFamily.gaussian],
        "student": [pv.BicopFamily.student],
        "archimedean": pv.archimedean,
        "tll": pv.tll,
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(
        f"Unknown family set '{name}'. Choose from: {list(mapping.keys())}"
    )
