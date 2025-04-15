"""Tests for clustering.core.schemas module."""

import pandas as pd
import polars as pl
import pytest
from pandera.errors import SchemaError

from clustering.core.schemas import (
    DistributedDataSchema,
    MergedDataSchema,
    NSMappingSchema,
    SalesSchema,
    Schema,
)


class TestSchema:
    """Tests for the base Schema class."""

    def test_check_method_pandas(self) -> None:
        """Test the check method with pandas DataFrame."""

        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Valid data
        valid_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = TestSchema.check(valid_data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

    def test_check_method_polars(self) -> None:
        """Test the check method with polars DataFrame."""

        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Valid data
        valid_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = TestSchema.check(valid_data)

        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

    def test_invalid_data(self) -> None:
        """Test that the schema raises an error for invalid data."""

        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Invalid data (wrong type)
        invalid_data = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3]})

        with pytest.raises(SchemaError):
            TestSchema.check(invalid_data)


class TestSalesSchema:
    """Tests for SalesSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid sales data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = SalesSchema.check(data)
        assert len(validated_data) == 3

    def test_data_coercion(self) -> None:
        """Test coercion of data types."""
        # Strings that should be coerced to numbers
        data = pd.DataFrame(
            {
                "SKU_NBR": ["101", "102", "103"],
                "STORE_NBR": ["1", "2", "3"],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "TOTAL_SALES": ["100.0", "200.0", "300.0"],
            }
        )

        validated_data = SalesSchema.check(data)
        assert isinstance(validated_data["SKU_NBR"].iloc[0], int)
        assert isinstance(validated_data["TOTAL_SALES"].iloc[0], float)

    def test_invalid_data_negative_values(self) -> None:
        """Test validation fails with negative SKU numbers."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [-1, 102, 103],  # Negative value, should fail
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        with pytest.raises(SchemaError):
            SalesSchema.check(data)

    def test_missing_required_column(self) -> None:
        """Test validation fails with missing columns."""
        # Missing TOTAL_SALES column
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
            }
        )

        with pytest.raises(SchemaError):
            SalesSchema.check(data)

    def test_empty_dataframe(self) -> None:
        """Test validation fails with empty DataFrame."""
        # Create empty DataFrame with correct columns
        data = pd.DataFrame({"SKU_NBR": [], "STORE_NBR": [], "CAT_DSC": [], "TOTAL_SALES": []})

        with pytest.raises(SchemaError):
            SalesSchema.check(data)


class TestNSMappingSchema:
    """Tests for NSMappingSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid need state mapping data."""
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3

    def test_data_coercion(self) -> None:
        """Test coercion of boolean values."""
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [1, 0, 1],  # Integers should be coerced to booleans
                "TO_BE_DROPPED": [0, 1, 0],  # Integers should be coerced to booleans
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert all(isinstance(x, bool) for x in validated_data["NEW_ITEM"])
        assert all(isinstance(x, bool) for x in validated_data["TO_BE_DROPPED"])

    def test_nullable_attributes(self) -> None:
        """Test that attributes can be null."""
        # All attributes are null
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": [None, None, None],
                "ATTRIBUTE_2": [None, None, None],
                "ATTRIBUTE_3": [None, None, None],
                "ATTRIBUTE_4": [None, None, None],
                "ATTRIBUTE_5": [None, None, None],
                "ATTRIBUTE_6": [None, None, None],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3


class TestMergedDataSchema:
    """Tests for MergedDataSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid merged data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = MergedDataSchema.check(data)
        assert len(validated_data) == 3

    def test_polars_dataframe(self) -> None:
        """Test that schema works with polars DataFrame."""
        data = pl.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = MergedDataSchema.check(data)
        assert isinstance(validated_data, pl.DataFrame)
        assert len(validated_data) == 3


class TestDistributedDataSchema:
    """Tests for DistributedDataSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid distributed data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Category A", "Category B", "Category C"],
                "NEED_STATE": ["State A", "State B", "State C"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = DistributedDataSchema.check(data)
        assert len(validated_data) == 3
