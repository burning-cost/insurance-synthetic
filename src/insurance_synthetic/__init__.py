"""
insurance-synthetic — vine copula synthetic insurance portfolio generator.

The problem this solves: actuarial teams often can't share real policyholder
data for model development, benchmarking, or vendor demos. Generic synthetic
data tools (SDV, CTGAN) work for generic tabular data, but they don't understand
insurance-specific structure: the exposure offset in frequency models, the zero-
inflation in severity columns, the tail dependence between risk factors.

This library generates synthetic portfolios that are actuarially valid:
- Claim counts respect the exposure offset (Poisson with lambda * exposure)
- Dependencies are captured by R-vine copulas — including tail dependence
  that Gaussian copulas miss (e.g. young driver + high vehicle group +
  zero NCD is more dangerous than the sum of marginal risks)
- Marginals are fitted automatically by AIC across Gamma, LogNormal, Poisson,
  NegBin, Normal, and Beta families
- Fidelity is quantified via KS statistics, Spearman correlation comparison,
  TVaR ratio, and TSTR (Train-on-Synthetic, Test-on-Real) Gini gap

Quick start::

    import polars as pl
    from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport

    synth = InsuranceSynthesizer()
    synth.fit(real_df, exposure_col='exposure', frequency_col='claim_count')
    synthetic_df = synth.generate(50_000)

    report = SyntheticFidelityReport(real_df, synthetic_df)
    print(report.to_markdown())

For a pre-built UK motor schema::

    from insurance_synthetic import uk_motor_schema
    schema = uk_motor_schema()

"""

from ._synthesiser import InsuranceSynthesizer
from ._marginals import FittedMarginal, fit_marginal
from ._fidelity import SyntheticFidelityReport
from ._schemas import uk_motor_schema, uk_employer_liability_schema, ColumnSpec

__all__ = [
    "InsuranceSynthesizer",
    "FittedMarginal",
    "fit_marginal",
    "SyntheticFidelityReport",
    "uk_motor_schema",
    "uk_employer_liability_schema",
    "ColumnSpec",
]

__version__ = "0.1.0"
