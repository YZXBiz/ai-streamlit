"""Tests for schema validation in the shared package."""


import numpy as np
import pandas as pd
import polars as pl
import pytest

from clustering.shared.schemas import (
    DataFrameType,
    DistributedDataSchema,
    MergedDataSchema,
    NSMappingSchema,
    SalesSchema,
    Schema,
)


class TestBaseSchema:
    """Tests for the base Schema class."""

    def test_schema_validation(self) -> None:
        """Test that Schema class can validate data."""

        # Define a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Valid data
        valid_pd_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        valid_pl_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Validate with pandas
        result_pd = TestSchema.check(valid_pd_data)
        assert isinstance(result_pd, pd.DataFrame)
        assert len(result_pd) == 3
        assert list(result_pd.columns) == ["col1", "col2"]

        # Validate with polars
        result_pl = TestSchema.check(valid_pl_data)
        assert isinstance(result_pl, pl.DataFrame)
        assert len(result_pl) == 3
        assert list(result_pl.columns) == ["col1", "col2"]

    def test_schema_validation_failure(self) -> None:
        """Test schema validation failure with invalid data."""

        # Define a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Invalid data - wrong type
        invalid_type_data = pd.DataFrame({"col1": ["1", "2", "3"], "col2": ["a", "b", "c"]})

        # Should raise an error due to type mismatch
        with pytest.raises(ValueError):
            TestSchema.check(invalid_type_data)

        # Invalid data - missing column
        invalid_columns_data = pd.DataFrame({"col1": [1, 2, 3]})

        # Should raise an error due to missing column
        with pytest.raises(ValueError):
            TestSchema.check(invalid_columns_data)


class TestSalesSchema:
    """Tests for the SalesSchema."""

    def test_valid_sales_data_pandas(self) -> None:
        """Test validation of valid sales data with pandas."""
        valid_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        result = SalesSchema.check(valid_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"]

    def test_valid_sales_data_polars(self) -> None:
        """Test validation of valid sales data with polars."""
        valid_data = pl.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        result = SalesSchema.check(valid_data)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result.columns == ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"]

    def test_sales_automatic_type_conversion(self) -> None:
        """Test automatic type conversion in SalesSchema."""
        # Data with types that need conversion
        data_to_convert = pd.DataFrame(
            {
                "SKU_NBR": ["1001", "1002", "1003"],  # strings should be converted to int
                "STORE_NBR": [101.0, 102.0, 103.0],  # floats should be converted to int
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [
                    "1500.50",
                    "2200.75",
                    "3100.25",
                ],  # strings should be converted to float
            }
        )

        result = SalesSchema.check(data_to_convert)
        assert result["SKU_NBR"].dtype == np.int64
        assert result["STORE_NBR"].dtype == np.int64
        assert result["TOTAL_SALES"].dtype == np.float64

    def test_sales_invalid_negative_values(self) -> None:
        """Test validation fails with negative values in fields that should be positive."""
        # Data with negative values
        invalid_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, -1002, 1003],  # negative value should fail
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        with pytest.raises(ValueError):
            SalesSchema.check(invalid_data)

        # Test negative sales
        invalid_sales = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, -2200.75, 3100.25],  # negative value should fail
            }
        )

        with pytest.raises(ValueError):
            SalesSchema.check(invalid_sales)

    def test_sales_null_values(self) -> None:
        """Test validation fails with null values."""
        # Data with null values
        invalid_null_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, None, 1003],  # None should fail
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        with pytest.raises(ValueError):
            SalesSchema.check(invalid_null_data)

    def test_sales_empty_dataframe(self) -> None:
        """Test validation fails with empty DataFrame."""
        # Empty DataFrame
        empty_data = pd.DataFrame(
            {"SKU_NBR": [], "STORE_NBR": [], "CAT_DSC": [], "TOTAL_SALES": []}
        )

        with pytest.raises(ValueError):
            SalesSchema.check(empty_data)


class TestNSMappingSchema:
    """Tests for the NSMappingSchema."""

    @pytest.fixture
    def valid_mapping_data(self) -> pd.DataFrame:
        """Fixture for valid NS mapping data."""
        return pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "CDT": ["Tablets", "Lotion", "Chips"],
                "ATTRIBUTE_1": ["OTC", "Natural", "Savory"],
                "ATTRIBUTE_2": ["Fast acting", "Hydrating", "Crunchy"],
                "ATTRIBUTE_3": [None, "Anti-aging", None],
                "ATTRIBUTE_4": ["Headache", None, None],
                "ATTRIBUTE_5": [None, None, "Party size"],
                "ATTRIBUTE_6": [None, "Fragrance-free", None],
                "PLANOGRAM_DSC": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                "PLANOGRAM_NBR": [10, 20, 30],
                "NEW_ITEM": [False, True, False],
                "TO_BE_DROPPED": [False, False, True],
            }
        )

    def test_valid_mapping_data(self, valid_mapping_data: pd.DataFrame) -> None:
        """Test validation of valid mapping data."""
        result = NSMappingSchema.check(valid_mapping_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "PRODUCT_ID" in result.columns
        assert "NEED_STATE" in result.columns

    def test_mapping_automatic_type_conversion(self) -> None:
        """Test automatic type conversion in NSMappingSchema."""
        # Data with types that need conversion
        data_to_convert = pd.DataFrame(
            {
                "PRODUCT_ID": ["101", "102", "103"],  # strings to int
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "CDT": ["Tablets", "Lotion", "Chips"],
                "ATTRIBUTE_1": ["OTC", "Natural", "Savory"],
                "ATTRIBUTE_2": ["Fast acting", "Hydrating", "Crunchy"],
                "ATTRIBUTE_3": [None, "Anti-aging", None],
                "ATTRIBUTE_4": ["Headache", None, None],
                "ATTRIBUTE_5": [None, None, "Party size"],
                "ATTRIBUTE_6": [None, "Fragrance-free", None],
                "PLANOGRAM_DSC": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                "PLANOGRAM_NBR": ["10", "20", "30"],  # strings to int
                "NEW_ITEM": [0, 1, 0],  # int to bool
                "TO_BE_DROPPED": ["False", "False", "True"],  # strings to bool
            }
        )

        result = NSMappingSchema.check(data_to_convert)
        assert result["PRODUCT_ID"].dtype == np.int64
        assert result["PLANOGRAM_NBR"].dtype == np.int64
        assert result["NEW_ITEM"].dtype == bool
        assert result["TO_BE_DROPPED"].dtype == bool

    def test_mapping_invalid_product_id(self) -> None:
        """Test validation fails with invalid product ID."""
        invalid_data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, -102, 103],  # negative value should fail
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "CDT": ["Tablets", "Lotion", "Chips"],
                "ATTRIBUTE_1": ["OTC", "Natural", "Savory"],
                "ATTRIBUTE_2": ["Fast acting", "Hydrating", "Crunchy"],
                "ATTRIBUTE_3": [None, "Anti-aging", None],
                "ATTRIBUTE_4": ["Headache", None, None],
                "ATTRIBUTE_5": [None, None, "Party size"],
                "ATTRIBUTE_6": [None, "Fragrance-free", None],
                "PLANOGRAM_DSC": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                "PLANOGRAM_NBR": [10, 20, 30],
                "NEW_ITEM": [False, True, False],
                "TO_BE_DROPPED": [False, False, True],
            }
        )

        with pytest.raises(ValueError):
            NSMappingSchema.check(invalid_data)

    def test_mapping_required_field_missing(self) -> None:
        """Test validation fails when required fields are missing."""
        missing_field_data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                # CATEGORY missing
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "CDT": ["Tablets", "Lotion", "Chips"],
                "ATTRIBUTE_1": ["OTC", "Natural", "Savory"],
                "ATTRIBUTE_2": ["Fast acting", "Hydrating", "Crunchy"],
                "ATTRIBUTE_3": [None, "Anti-aging", None],
                "ATTRIBUTE_4": ["Headache", None, None],
                "ATTRIBUTE_5": [None, None, "Party size"],
                "ATTRIBUTE_6": [None, "Fragrance-free", None],
                "PLANOGRAM_DSC": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                "PLANOGRAM_NBR": [10, 20, 30],
                "NEW_ITEM": [False, True, False],
                "TO_BE_DROPPED": [False, False, True],
            }
        )

        with pytest.raises(ValueError):
            NSMappingSchema.check(missing_field_data)


class TestMergedDataSchema:
    """Tests for the MergedDataSchema."""

    def test_valid_merged_data(self) -> None:
        """Test validation of valid merged data."""
        valid_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        result = MergedDataSchema.check(valid_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == [
            "SKU_NBR",
            "STORE_NBR",
            "CAT_DSC",
            "NEED_STATE",
            "TOTAL_SALES",
        ]

    def test_merged_invalid_data(self) -> None:
        """Test validation fails with invalid merged data."""
        # Missing required field
        missing_field = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                # NEED_STATE missing
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        with pytest.raises(ValueError):
            MergedDataSchema.check(missing_field)

        # Invalid value type
        invalid_type = pd.DataFrame(
            {
                "SKU_NBR": ["1001", "1002", "1003"],  # strings, will be coerced
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "TOTAL_SALES": ["not a number", "2200.75", "3100.25"],  # invalid value
            }
        )

        with pytest.raises(ValueError):
            MergedDataSchema.check(invalid_type)


class TestDistributedDataSchema:
    """Tests for the DistributedDataSchema."""

    def test_valid_distributed_data(self) -> None:
        """Test validation of valid distributed data."""
        valid_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        result = DistributedDataSchema.check(valid_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == [
            "SKU_NBR",
            "STORE_NBR",
            "CAT_DSC",
            "NEED_STATE",
            "TOTAL_SALES",
        ]

    def test_distributed_edge_cases(self) -> None:
        """Test edge cases for distributed data schema."""
        # Zero sales should be valid
        zero_sales = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "TOTAL_SALES": [0.0, 0.0, 0.0],
            }
        )

        result = DistributedDataSchema.check(zero_sales)
        assert len(result) == 3
        assert all(result["TOTAL_SALES"] == 0.0)

        # Very large values should be valid
        large_values = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "TOTAL_SALES": [1e10, 2e10, 3e10],  # Very large values
            }
        )

        result = DistributedDataSchema.check(large_values)
        assert len(result) == 3
        assert all(result["TOTAL_SALES"] >= 1e10)


class TestDataTypeConversions:
    """Tests for data type conversions across schemas."""

    def test_dataframe_type_alias(self) -> None:
        """Test DataFrameType type alias works with different data types."""
        # Create test data in different formats
        pd_data = pd.DataFrame({"col1": [1, 2, 3]})
        pl_data = pl.DataFrame({"col1": [1, 2, 3]})
        np_data = np.array([[1], [2], [3]])

        # Function that accepts DataFrameType
        def process_data(data: DataFrameType) -> int:
            if isinstance(data, pd.DataFrame):
                return int(data["col1"].sum())
            elif isinstance(data, pl.DataFrame):
                return int(data["col1"].sum())
            elif isinstance(data, np.ndarray):
                return int(data.sum())
            raise TypeError(f"Unsupported type: {type(data)}")

        # Test with each type
        assert process_data(pd_data) == 6
        assert process_data(pl_data) == 6
        assert process_data(np_data) == 6
