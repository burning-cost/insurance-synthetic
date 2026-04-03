"""Extended tests for schema definitions — ColumnSpec and schema factory functions."""

import pytest
import polars as pl

from insurance_synthetic._schemas import ColumnSpec, uk_motor_schema, uk_employer_liability_schema


# ---------------------------------------------------------------------------
# ColumnSpec
# ---------------------------------------------------------------------------

class TestColumnSpecExtended:
    def test_is_severity_false_by_default(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.is_severity is False

    def test_is_frequency_false_by_default(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.is_frequency is False

    def test_all_flags_set(self):
        spec = ColumnSpec(
            name="claim_count",
            dtype="int",
            is_target=True,
            is_frequency=True,
        )
        assert spec.is_target
        assert spec.is_frequency

    def test_categories_default_empty(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.categories == []

    def test_categories_not_shared_between_instances(self):
        # Mutable defaults should not be shared
        s1 = ColumnSpec(name="a", dtype="categorical")
        s2 = ColumnSpec(name="b", dtype="categorical")
        s1.categories.append("foo")
        assert "foo" not in s2.categories

    def test_description_default_empty_string(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.description == ""

    def test_min_val_max_val_none_by_default(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.min_val is None
        assert spec.max_val is None

    def test_string_dtype_valid(self):
        spec = ColumnSpec(name="region", dtype="categorical",
                          categories=["London", "North"])
        assert spec.dtype == "categorical"

    def test_bool_dtype(self):
        spec = ColumnSpec(name="telematics", dtype="bool")
        assert spec.dtype == "bool"


# ---------------------------------------------------------------------------
# UK Motor schema — deeper validation
# ---------------------------------------------------------------------------

class TestUKMotorSchemaExtended:
    def setup_method(self):
        self.schema = uk_motor_schema()
        self.col_map = {c.name: c for c in self.schema["columns"]}

    def test_driver_age_bounds(self):
        spec = self.col_map["driver_age"]
        assert spec.min_val == 17
        assert spec.max_val == 90

    def test_vehicle_age_min_zero(self):
        spec = self.col_map["vehicle_age"]
        assert spec.min_val == 0

    def test_vehicle_group_range(self):
        spec = self.col_map["vehicle_group"]
        assert spec.min_val == 1
        assert spec.max_val == 50

    def test_exposure_min_positive(self):
        spec = self.col_map["exposure"]
        assert spec.min_val > 0

    def test_exposure_max_one(self):
        spec = self.col_map["exposure"]
        assert spec.max_val == 1.0

    def test_exposure_is_exposure_flag(self):
        spec = self.col_map["exposure"]
        assert spec.is_exposure

    def test_claim_count_is_target_and_frequency(self):
        spec = self.col_map["claim_count"]
        assert spec.is_target
        assert spec.is_frequency
        assert not spec.is_severity

    def test_claim_amount_is_target_and_severity(self):
        spec = self.col_map["claim_amount"]
        assert spec.is_target
        assert spec.is_severity
        assert not spec.is_frequency

    def test_region_categories_count(self):
        spec = self.col_map["region"]
        assert len(spec.categories) >= 8  # At least 8 UK regions

    def test_cover_type_three_levels(self):
        spec = self.col_map["cover_type"]
        assert len(spec.categories) == 3

    def test_payment_method_categories(self):
        spec = self.col_map["payment_method"]
        assert "Annual" in spec.categories

    def test_ncd_years_max_25(self):
        spec = self.col_map["ncd_years"]
        assert spec.max_val == 25

    def test_annual_mileage_min(self):
        spec = self.col_map["annual_mileage"]
        assert spec.min_val >= 1000

    def test_annual_mileage_max(self):
        spec = self.col_map["annual_mileage"]
        assert spec.max_val <= 100_000

    def test_constraints_include_exposure(self):
        assert "exposure" in self.schema["constraints"]

    def test_constraints_include_claim_count(self):
        assert "claim_count" in self.schema["constraints"]

    def test_constraints_claim_count_non_negative(self):
        lo, hi = self.schema["constraints"]["claim_count"]
        assert lo == 0
        assert hi > 0

    def test_constraints_claim_amount_non_negative(self):
        lo, hi = self.schema["constraints"]["claim_amount"]
        assert lo == 0.0

    def test_description_mentions_motor(self):
        desc = self.schema["description"].lower()
        assert "motor" in desc or "vehicle" in desc

    def test_all_constraints_have_lo_le_hi(self):
        for col, (lo, hi) in self.schema["constraints"].items():
            assert lo <= hi, f"{col}: lo={lo} > hi={hi}"


# ---------------------------------------------------------------------------
# UK Employer Liability schema — deeper validation
# ---------------------------------------------------------------------------

class TestUKELSchemaExtended:
    def setup_method(self):
        self.schema = uk_employer_liability_schema()
        self.col_map = {c.name: c for c in self.schema["columns"]}

    def test_employee_count_min_one(self):
        spec = self.col_map["employee_count"]
        assert spec.min_val == 1

    def test_payroll_min_positive(self):
        spec = self.col_map["payroll_gbp"]
        assert spec.min_val > 0

    def test_years_trading_min_zero(self):
        spec = self.col_map["years_trading"]
        assert spec.min_val == 0

    def test_claims_history_count_min_zero(self):
        spec = self.col_map["claims_history_count"]
        assert spec.min_val == 0

    def test_sic_division_is_categorical(self):
        spec = self.col_map["sic_division"]
        assert spec.dtype == "categorical"

    def test_exposure_flags(self):
        # EL has two exposure-like columns: payroll and exposure
        exposure_cols = [c for c in self.schema["columns"] if c.is_exposure]
        exposure_names = {c.name for c in exposure_cols}
        assert "payroll_gbp" in exposure_names or "exposure" in exposure_names

    def test_claim_amount_severity_flag(self):
        spec = self.col_map["claim_amount"]
        assert spec.is_severity

    def test_sic_categories_contain_construction(self):
        spec = self.col_map["sic_division"]
        construction = [c for c in spec.categories if "Construction" in c or "F -" in c]
        assert len(construction) > 0

    def test_all_constraints_valid(self):
        for col, (lo, hi) in self.schema["constraints"].items():
            assert lo <= hi, f"{col}: lo={lo} > hi={hi}"

    def test_description_mentions_liability(self):
        desc = self.schema["description"].lower()
        assert "liability" in desc or "el" in desc or "employer" in desc
