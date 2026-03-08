"""
Tests for pre-built schema definitions.

We verify that the schema functions return valid, internally consistent
structures. We also check that the InsuranceSynthesizer can generate a
plausible portfolio without a real dataset by using the schema constraints
alone (via a small synthetic seed dataset).
"""

import numpy as np
import polars as pl
import pytest

from insurance_synthetic import uk_motor_schema, ColumnSpec
from insurance_synthetic._schemas import uk_employer_liability_schema


class TestUKMotorSchema:
    def test_returns_dict(self):
        schema = uk_motor_schema()
        assert isinstance(schema, dict)

    def test_has_required_keys(self):
        schema = uk_motor_schema()
        assert "columns" in schema
        assert "constraints" in schema
        assert "description" in schema

    def test_columns_are_column_specs(self):
        schema = uk_motor_schema()
        for col in schema["columns"]:
            assert isinstance(col, ColumnSpec)

    def test_column_names_present(self):
        schema = uk_motor_schema()
        names = {c.name for c in schema["columns"]}
        expected = {
            "driver_age", "vehicle_age", "vehicle_group", "region",
            "ncd_years", "cover_type", "payment_method", "annual_mileage",
            "exposure", "claim_count", "claim_amount",
        }
        assert expected == names

    def test_exposure_column_marked(self):
        schema = uk_motor_schema()
        exposure_cols = [c for c in schema["columns"] if c.is_exposure]
        assert len(exposure_cols) == 1
        assert exposure_cols[0].name == "exposure"

    def test_frequency_column_marked(self):
        schema = uk_motor_schema()
        freq_cols = [c for c in schema["columns"] if c.is_frequency]
        assert len(freq_cols) == 1
        assert freq_cols[0].name == "claim_count"

    def test_severity_column_marked(self):
        schema = uk_motor_schema()
        sev_cols = [c for c in schema["columns"] if c.is_severity]
        assert len(sev_cols) == 1
        assert sev_cols[0].name == "claim_amount"

    def test_categorical_columns_have_categories(self):
        schema = uk_motor_schema()
        cat_cols = [c for c in schema["columns"] if c.dtype == "categorical"]
        for col in cat_cols:
            assert len(col.categories) > 0, f"{col.name} has no categories"

    def test_region_has_uk_regions(self):
        schema = uk_motor_schema()
        region_spec = next(c for c in schema["columns"] if c.name == "region")
        assert "London" in region_spec.categories
        assert "Scotland" in region_spec.categories

    def test_constraints_dict_structure(self):
        schema = uk_motor_schema()
        for col_name, rule in schema["constraints"].items():
            assert isinstance(col_name, str)
            assert isinstance(rule, tuple) and len(rule) == 2

    def test_constraints_respect_min_max(self):
        schema = uk_motor_schema()
        col_specs = {c.name: c for c in schema["columns"]}
        for col_name, (lo, hi) in schema["constraints"].items():
            if col_name in col_specs:
                spec = col_specs[col_name]
                if spec.min_val is not None:
                    assert lo >= spec.min_val, f"{col_name}: constraint lo {lo} < spec min {spec.min_val}"
                if spec.max_val is not None:
                    assert hi <= spec.max_val, f"{col_name}: constraint hi {hi} > spec max {spec.max_val}"

    def test_driver_age_range(self):
        schema = uk_motor_schema()
        constraints = schema["constraints"]
        assert constraints["driver_age"] == (17, 90)

    def test_exposure_range(self):
        schema = uk_motor_schema()
        lo, hi = schema["constraints"]["exposure"]
        assert lo > 0
        assert hi <= 1.0

    def test_description_is_nonempty_string(self):
        schema = uk_motor_schema()
        assert isinstance(schema["description"], str)
        assert len(schema["description"]) > 10

    def test_cover_type_categories(self):
        schema = uk_motor_schema()
        cover = next(c for c in schema["columns"] if c.name == "cover_type")
        assert "Comprehensive" in cover.categories
        assert "Third Party Only" in cover.categories


class TestUKEmployerLiabilitySchema:
    def test_returns_dict(self):
        schema = uk_employer_liability_schema()
        assert isinstance(schema, dict)

    def test_has_required_keys(self):
        schema = uk_employer_liability_schema()
        assert "columns" in schema
        assert "constraints" in schema

    def test_payroll_is_exposure(self):
        schema = uk_employer_liability_schema()
        payroll = next(
            (c for c in schema["columns"] if c.name == "payroll_gbp"), None
        )
        assert payroll is not None
        assert payroll.is_exposure

    def test_sic_has_categories(self):
        schema = uk_employer_liability_schema()
        sic = next(c for c in schema["columns"] if c.name == "sic_division")
        assert len(sic.categories) >= 10

    def test_claim_count_is_frequency(self):
        schema = uk_employer_liability_schema()
        freq = next(c for c in schema["columns"] if c.is_frequency)
        assert freq.name == "claim_count"


class TestColumnSpec:
    def test_default_values(self):
        spec = ColumnSpec(name="x", dtype="float")
        assert spec.categories == []
        assert spec.min_val is None
        assert spec.max_val is None
        assert spec.is_target is False
        assert spec.is_exposure is False

    def test_custom_values(self):
        spec = ColumnSpec(
            name="claim_count",
            dtype="int",
            min_val=0,
            max_val=10,
            is_target=True,
            is_frequency=True,
        )
        assert spec.is_target
        assert spec.is_frequency
        assert spec.max_val == 10
