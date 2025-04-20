"""Tests for internal preprocessing assets."""

import polars as pl
import pytest

from clustering.pipeline.assets.preprocessing.internal import (
    internal_normalized_sales_data,
    internal_output_sales_table,
    internal_product_category_mapping,
    internal_raw_sales_data,
    internal_sales_by_category,
    internal_sales_with_categories,
)


@pytest.fixture
def mock_raw_sales_data() -> pl.DataFrame:
    """Create mock raw sales data for testing."""
    return pl.DataFrame(
        {
            "SKU_NBR": [1001, 1002, 1001, 1003, 1002],
            "STORE_NBR": [201, 201, 202, 202, 203],
            "TOTAL_SALES": [100.0, 200.0, 150.0, 300.0, 250.0],
            "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"],
        }
    )


@pytest.fixture
def mock_category_mapping() -> pl.DataFrame:
    """Create mock product category mapping for testing."""
    return pl.DataFrame(
        {
            "PRODUCT_ID": [1001, 1002, 1003],
            "CATEGORY": ["Grocery", "Beverages", "Grocery"],
            "NEED_STATE": ["CONVENIENCE_1", "CONVENIENCE_2", "CONVENIENCE_1"],
            "CDT": ["Convenience_1", "Convenience_2", "Convenience_1"],
        }
    )


class TestInternalRawSalesData:
    """Tests for internal_raw_sales_data asset."""

    def test_load_raw_data(self, mock_execution_context):
        """Test loading raw sales data from resource."""
        # Configure the mock reader to return test data
        mock_data = pl.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [201, 202, 203],
                "CAT_DSC": ["Grocery", "Beverages", "Grocery"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )
        # Set data on the existing MockReader
        mock_execution_context.resources.sales_data_reader.data = mock_data

        # Execute the asset
        result = internal_raw_sales_data(mock_execution_context)

        # Verify result is the data from the reader
        assert isinstance(result, pl.DataFrame)
        assert result.equals(mock_data)


class TestInternalProductCategoryMapping:
    """Tests for internal_product_category_mapping asset."""

    def test_load_category_mapping(self, mock_execution_context):
        """Test loading product category mapping from resource."""
        # Configure the mock reader to return test data
        mock_data = pl.DataFrame(
            {
                "PRODUCT_ID": [1001, 1002, 1003],
                "CATEGORY": ["Grocery", "Beverages", "Grocery"],
                "NEED_STATE": ["CONVENIENCE_1", "CONVENIENCE_2", "CONVENIENCE_1"],
                "CDT": ["Convenience_1", "Convenience_2", "Convenience_1"],
            }
        )
        # Set data on the existing MockReader
        mock_execution_context.resources.category_mapping_reader.data = mock_data

        # Execute the asset
        result = internal_product_category_mapping(mock_execution_context)

        # Verify result is the data from the reader
        assert isinstance(result, pl.DataFrame)
        assert result.equals(mock_data)


class TestInternalSalesWithCategories:
    """Tests for internal_sales_with_categories asset."""

    def test_join_sales_with_categories(
        self, mock_execution_context, mock_raw_sales_data, mock_category_mapping
    ):
        """Test joining sales data with category mapping."""
        # Modify test data to ensure it matches the expected format
        sales_data = pl.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [201, 202, 203],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )
        category_data = pl.DataFrame(
            {
                "PRODUCT_ID": [1001, 1002, 1003],
                "CATEGORY": ["Grocery", "Beverages", "Grocery"],
                "NEED_STATE": ["CONVENIENCE_1", "CONVENIENCE_2", "CONVENIENCE_1"],
            }
        )

        # No need to replace readers, the asset takes dataframes as input
        # mock_execution_context.resources.sales_data_reader = MagicMock() # Remove this
        # mock_execution_context.resources.category_mapping_reader = MagicMock() # Remove this

        # Execute the asset
        result = internal_sales_with_categories(mock_execution_context, sales_data, category_data)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "STORE_NBR" in result.columns
        assert "CAT_DSC" in result.columns  # Asset renames CATEGORY to CAT_DSC
        assert "NEED_STATE" in result.columns
        assert "TOTAL_SALES" in result.columns

        # Verify join logic (example check)
        assert result.shape[0] == 3  # Check if all rows joined
        assert result.filter(pl.col("STORE_NBR") == 201)["CAT_DSC"][0] == "Grocery"


class TestInternalSalesByCategory:
    """Tests for internal_sales_by_category asset."""

    def test_group_sales_by_category(self, mock_execution_context):
        """Test grouping sales data by store and category."""
        # Create test data with correct column names
        sales_with_categories = pl.DataFrame(
            {
                "STORE_NBR": [201, 201, 202, 202, 203],
                "SKU_NBR": [1001, 1002, 1001, 1003, 1002],
                "CAT_DSC": ["Grocery", "Beverages", "Grocery", "Grocery", "Beverages"],
                "NEED_STATE": [
                    "CONVENIENCE_1",
                    "CONVENIENCE_2",
                    "CONVENIENCE_1",
                    "CONVENIENCE_1",
                    "CONVENIENCE_2",
                ],
                "TOTAL_SALES": [100.0, 200.0, 150.0, 300.0, 250.0],
            }
        )

        # Execute the asset
        result = internal_sales_by_category(mock_execution_context, sales_with_categories)

        # Verify result structure
        assert isinstance(result, dict)
        # Should have entries for each category
        assert "Grocery" in result
        assert "Beverages" in result

        # Check data for at least one category
        grocery_data = result["Grocery"]
        assert isinstance(grocery_data, pl.DataFrame)
        assert "STORE_NBR" in grocery_data.columns


class TestInternalOutputSalesTable:
    """Tests for internal_output_sales_table asset."""

    def test_pivot_sales_table(self, mock_execution_context):
        """Test pivoting sales data to wide format."""
        # Create test data with aggregated sales
        sales_by_category = pl.DataFrame(
            {
                "STORE_NBR": [201, 201, 202, 202, 203],
                "CAT_DSC": ["Grocery", "Beverages", "Grocery", "Beverages", "Beverages"],
                "TOTAL_SALES": [100.0, 200.0, 450.0, 120.0, 250.0],
            }
        )

        # Execute the asset
        result = internal_output_sales_table(mock_execution_context, sales_by_category)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "STORE_NBR" in result.columns
        assert "Grocery" in result.columns
        assert "Beverages" in result.columns

        # Verify pivot worked correctly
        # Check store 201
        store201 = result.filter(pl.col("STORE_NBR") == 201)
        assert store201["Grocery"].to_list()[0] == 100.0
        assert store201["Beverages"].to_list()[0] == 200.0


class TestInternalNormalizedSalesData:
    """Tests for internal_normalized_sales_data asset."""

    def test_normalize_sales_data(self, mock_execution_context):
        """Test normalizing sales data."""
        # Create test data with required column format
        sales_data = pl.DataFrame(
            {
                "STORE_NBR": [201, 202, 203],
                "SKU_NBR": [1001, 1002, 1003],
                "CAT_DSC": ["Grocery", "Beverages", "Grocery"],
                "NEED_STATE": ["CONVENIENCE_1", "CONVENIENCE_2", "CONVENIENCE_1"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        # Configure normalization method in context
        mock_execution_context.resources.config.normalize_method = "min_max"

        # Execute the asset
        result = internal_normalized_sales_data(mock_execution_context, sales_data)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "STORE_NBR" in result.columns
        assert "SKU_NBR" in result.columns
        assert "CAT_DSC" in result.columns
        assert "NEED_STATE" in result.columns
        assert "TOTAL_SALES" in result.columns

        # Check that sales data is normalized
        min_sales = result["TOTAL_SALES"].min()
        max_sales = result["TOTAL_SALES"].max()
        assert min_sales >= 0.0
        assert max_sales <= 1.0
