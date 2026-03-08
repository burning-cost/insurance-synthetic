"""
Pre-built schema definitions for common insurance portfolio types.

A schema defines the column structure, realistic value ranges, and data types
for a class of business. The synthesiser uses schemas to know which columns
are categorical, which are discrete (counts), and what reasonable bounds look
like for post-generation constraint enforcement.

Why bother with a schema? Because the fit/generate pipeline doesn't know
that driver_age can't be 7 or ncd_years can't exceed driver_age - 17.
Schemas encode that domain knowledge cleanly, separate from the statistical
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ColumnSpec:
    """
    Specification for a single portfolio column.

    Attributes
    ----------
    name : str
    dtype : str
        'float', 'int', 'categorical', 'bool'
    categories : list, optional
        Valid categories for categorical columns (in order of plausibility).
    min_val : float, optional
        Hard lower bound used in constraint enforcement.
    max_val : float, optional
        Hard upper bound.
    description : str
        Human-readable description for documentation.
    is_target : bool
        True if this column is a modelling target (loss count, severity),
        rather than a rating factor.
    is_exposure : bool
        True if this column represents exposure / risk period.
    is_frequency : bool
        True if this column is a claim count to be modelled with exposure offset.
    is_severity : bool
        True if this column is a claim amount / severity column.
    """
    name: str
    dtype: str
    categories: list = field(default_factory=list)
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""
    is_target: bool = False
    is_exposure: bool = False
    is_frequency: bool = False
    is_severity: bool = False


def uk_motor_schema() -> dict[str, Any]:
    """
    Column definitions for a UK private motor portfolio.

    Returns a dict with keys:
    - 'columns': list of ColumnSpec
    - 'constraints': dict mapping column name -> (min, max) or callable
    - 'description': str

    Columns cover the typical rating factors used by UK personal lines pricing
    teams: driver demographics, vehicle characteristics, policy structure, and
    claims experience. Distributions reflect a mid-market book with a spread
    from young inexperienced drivers through to mature low-risk policyholders.

    The claim_count and claim_amount columns are the primary modelling targets.
    Exposure is measured in policy years (0, 1].
    """
    columns = [
        ColumnSpec(
            name="driver_age",
            dtype="int",
            min_val=17,
            max_val=90,
            description="Age of the main driver in years.",
        ),
        ColumnSpec(
            name="vehicle_age",
            dtype="int",
            min_val=0,
            max_val=25,
            description="Age of the vehicle in years at inception.",
        ),
        ColumnSpec(
            name="vehicle_group",
            dtype="int",
            min_val=1,
            max_val=50,
            description=(
                "ABI vehicle group (1-50). Higher groups are larger, "
                "more powerful, more expensive to repair."
            ),
        ),
        ColumnSpec(
            name="region",
            dtype="categorical",
            categories=[
                "London", "South East", "East of England", "South West",
                "West Midlands", "East Midlands", "Yorkshire", "North West",
                "North East", "Scotland", "Wales", "Northern Ireland",
            ],
            description="UK region of the policyholder's registered address.",
        ),
        ColumnSpec(
            name="ncd_years",
            dtype="int",
            min_val=0,
            max_val=25,
            description=(
                "No-claims discount years. Proxy for driving experience "
                "and claims-free history."
            ),
        ),
        ColumnSpec(
            name="cover_type",
            dtype="categorical",
            categories=["Comprehensive", "Third Party Fire & Theft", "Third Party Only"],
            description="Level of cover purchased.",
        ),
        ColumnSpec(
            name="payment_method",
            dtype="categorical",
            categories=["Annual", "Monthly Direct Debit", "Monthly Credit"],
            description=(
                "How the premium is paid. Monthly credit correlates with "
                "higher-risk policyholders in practice."
            ),
        ),
        ColumnSpec(
            name="annual_mileage",
            dtype="int",
            min_val=1000,
            max_val=50000,
            description="Estimated annual mileage at inception.",
        ),
        ColumnSpec(
            name="exposure",
            dtype="float",
            min_val=0.01,
            max_val=1.0,
            description=(
                "Exposure in policy years. Policies mid-term at a reporting "
                "date will have fractional exposure."
            ),
            is_exposure=True,
        ),
        ColumnSpec(
            name="claim_count",
            dtype="int",
            min_val=0,
            max_val=10,
            description="Number of at-fault claims in the policy period.",
            is_target=True,
            is_frequency=True,
        ),
        ColumnSpec(
            name="claim_amount",
            dtype="float",
            min_val=0.0,
            max_val=500_000.0,
            description=(
                "Total incurred claim amount for the policy period. "
                "Zero for policies with no claims."
            ),
            is_target=True,
            is_severity=True,
        ),
    ]

    constraints: dict[str, Any] = {
        "driver_age": (17, 90),
        "vehicle_age": (0, 25),
        "vehicle_group": (1, 50),
        "ncd_years": (0, 25),
        "annual_mileage": (1000, 50000),
        "exposure": (0.01, 1.0),
        "claim_count": (0, 10),
        "claim_amount": (0.0, 500_000.0),
    }

    return {
        "columns": columns,
        "constraints": constraints,
        "description": (
            "UK private motor portfolio schema. "
            "Rating factors: driver_age, vehicle_age, vehicle_group, region, "
            "ncd_years, cover_type, payment_method, annual_mileage. "
            "Targets: claim_count (frequency), claim_amount (severity). "
            "Exposure: policy years in (0, 1]."
        ),
    }


def uk_employer_liability_schema() -> dict[str, Any]:
    """
    Column definitions for a UK employer's liability (EL) portfolio.

    A simpler schema than motor — fewer rating factors, but with important
    structural features: payroll as the exposure base, SIC code as a categorical
    with many levels, and claims that are typically more severe but rarer.
    """
    columns = [
        ColumnSpec(
            name="sic_division",
            dtype="categorical",
            categories=[
                "A - Agriculture", "B - Mining", "C - Manufacturing",
                "D - Utilities", "E - Water", "F - Construction",
                "G - Wholesale/Retail", "H - Transport", "I - Accommodation",
                "J - Information", "K - Finance", "L - Real Estate",
                "M - Professional", "N - Administrative", "P - Education",
                "Q - Health", "R - Arts", "S - Other Services",
            ],
            description="Broad SIC division of the insured business.",
        ),
        ColumnSpec(
            name="employee_count",
            dtype="int",
            min_val=1,
            max_val=5000,
            description="Number of full-time-equivalent employees.",
        ),
        ColumnSpec(
            name="payroll_gbp",
            dtype="float",
            min_val=10_000.0,
            max_val=50_000_000.0,
            description="Annual payroll in GBP. Used as the exposure base.",
            is_exposure=True,
        ),
        ColumnSpec(
            name="years_trading",
            dtype="int",
            min_val=0,
            max_val=100,
            description="Number of years the business has been trading.",
        ),
        ColumnSpec(
            name="claims_history_count",
            dtype="int",
            min_val=0,
            max_val=20,
            description="Number of EL claims in the prior 5 years.",
        ),
        ColumnSpec(
            name="exposure",
            dtype="float",
            min_val=0.01,
            max_val=1.0,
            description="Exposure in policy years.",
            is_exposure=True,
        ),
        ColumnSpec(
            name="claim_count",
            dtype="int",
            min_val=0,
            max_val=15,
            description="Number of EL claims in the policy period.",
            is_target=True,
            is_frequency=True,
        ),
        ColumnSpec(
            name="claim_amount",
            dtype="float",
            min_val=0.0,
            max_val=10_000_000.0,
            description="Total incurred claim amount for the policy period.",
            is_target=True,
            is_severity=True,
        ),
    ]

    constraints: dict[str, Any] = {
        "employee_count": (1, 5000),
        "payroll_gbp": (10_000.0, 50_000_000.0),
        "years_trading": (0, 100),
        "claims_history_count": (0, 20),
        "exposure": (0.01, 1.0),
        "claim_count": (0, 15),
        "claim_amount": (0.0, 10_000_000.0),
    }

    return {
        "columns": columns,
        "constraints": constraints,
        "description": (
            "UK employer's liability portfolio schema. "
            "Rating factors: sic_division, employee_count, payroll_gbp, "
            "years_trading, claims_history_count. "
            "Targets: claim_count (frequency), claim_amount (severity). "
            "Exposure: policy years in (0, 1]."
        ),
    }
