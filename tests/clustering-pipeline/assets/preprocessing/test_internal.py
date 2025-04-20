"""Tests for internal preprocessing assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from clustering.pipeline.assets.preprocessing.internal import (
    internal_raw_sales_data,
    internal_product_category_mapping,
    internal_sales_with_categories,
    internal_sales_by_category,
    internal_output_sales_table,
    internal_normalized_sales_data,
)


@pytest.fixture
def mock_raw_sales_data() -> pl.DataFrame:
    """Create mock raw sales data for testing."""
    return pl.DataFrame({
        "store_id": ["store_1", "store_1", "store_2", "store_2", "store_3"],
        "product_id": ["prod_1", "prod_2", "prod_1", "prod_3", "prod_2"],
        "sales_amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"],
    })


@pytest.fixture
def mock_category_mapping() -> pl.DataFrame:
    """Create mock product category mapping for testing."""
    return pl.DataFrame({
        "product_id": ["prod_1", "prod_2", "prod_3"],
        "category": ["category_a", "category_b", "category_a"],
    })


class TestInternalRawSalesData:
    """Tests for internal_raw_sales_data asset."""
    
    def test_load_raw_data(self, mock_execution_context):
        """Test loading raw sales data from resource."""
        # Configure the mock reader to return test data
        mock_data = pl.DataFrame({
            "store_id": ["store_1", "store_2", "store_3"],
            "product_id": ["prod_1", "prod_2", "prod_3"],
            "sales_amount": [100.0, 200.0, 300.0],
        })
        mock_execution_context.resources.sales_data_reader = MagicMock()
        mock_execution_context.resources.sales_data_reader.read.return_value = mock_data
        
        # Execute the asset
        result = internal_raw_sales_data(mock_execution_context)
        
        # Verify result is the data from the reader
        assert isinstance(result, pl.DataFrame)
        assert result.equals(mock_data)
        
        # Verify reader was called
        mock_execution_context.resources.sales_data_reader.read.assert_called_once()


class TestInternalProductCategoryMapping:
    """Tests for internal_product_category_mapping asset."""
    
    def test_load_category_mapping(self, mock_execution_context):
        """Test loading product category mapping from resource."""
        # Configure the mock reader to return test data
        mock_data = pl.DataFrame({
            "product_id": ["prod_1", "prod_2", "prod_3"],
            "category": ["category_a", "category_b", "category_a"],
        })
        mock_execution_context.resources.category_mapping_reader = MagicMock()
        mock_execution_context.resources.category_mapping_reader.read.return_value = mock_data
        
        # Execute the asset
        result = internal_product_category_mapping(mock_execution_context)
        
        # Verify result is the data from the reader
        assert isinstance(result, pl.DataFrame)
        assert result.equals(mock_data)
        
        # Verify reader was called
        mock_execution_context.resources.category_mapping_reader.read.assert_called_once()


class TestInternalSalesWithCategories:
    """Tests for internal_sales_with_categories asset."""
    
    def test_join_sales_with_categories(self, mock_execution_context, mock_raw_sales_data, mock_category_mapping):
        """Test joining sales data with category mapping."""
        # Execute the asset
        result = internal_sales_with_categories(
            mock_execution_context,
            mock_raw_sales_data,
            mock_category_mapping
        )
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "product_id" in result.columns
        assert "sales_amount" in result.columns
        assert "category" in result.columns
        
        # Verify all sales records have category information
        assert not result["category"].is_null().any()
        
        # Verify data was joined correctly
        assert result.filter(pl.col("product_id") == "prod_1")["category"].to_list()[0] == "category_a"
        assert result.filter(pl.col("product_id") == "prod_2")["category"].to_list()[0] == "category_b"


class TestInternalSalesByCategory:
    """Tests for internal_sales_by_category asset."""
    
    def test_group_sales_by_category(self, mock_execution_context):
        """Test grouping sales data by store and category."""
        # Create test data with categories
        sales_with_categories = pl.DataFrame({
            "store_id": ["store_1", "store_1", "store_2", "store_2", "store_3"],
            "product_id": ["prod_1", "prod_2", "prod_1", "prod_3", "prod_2"],
            "sales_amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            "category": ["category_a", "category_b", "category_a", "category_a", "category_b"],
        })
        
        # Execute the asset
        result = internal_sales_by_category(mock_execution_context, sales_with_categories)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "category" in result.columns
        assert "total_sales" in result.columns
        
        # Verify aggregation by store and category
        # Store 1, category_a should have sales of 100.0
        store1_cat_a = result.filter((pl.col("store_id") == "store_1") & (pl.col("category") == "category_a"))
        assert store1_cat_a["total_sales"].to_list()[0] == 100.0
        
        # Store 2, category_a should have sales of 150.0 + 300.0 = 450.0
        store2_cat_a = result.filter((pl.col("store_id") == "store_2") & (pl.col("category") == "category_a"))
        assert store2_cat_a["total_sales"].to_list()[0] == 450.0


class TestInternalOutputSalesTable:
    """Tests for internal_output_sales_table asset."""
    
    def test_pivot_sales_table(self, mock_execution_context):
        """Test pivoting sales data to wide format."""
        # Create test data with aggregated sales
        sales_by_category = pl.DataFrame({
            "store_id": ["store_1", "store_1", "store_2", "store_2", "store_3"],
            "category": ["category_a", "category_b", "category_a", "category_b", "category_b"],
            "total_sales": [100.0, 200.0, 450.0, 120.0, 250.0],
        })
        
        # Execute the asset
        result = internal_output_sales_table(mock_execution_context, sales_by_category)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "category_a" in result.columns
        assert "category_b" in result.columns
        
        # Verify pivot worked correctly
        # Check store_1
        store1 = result.filter(pl.col("store_id") == "store_1")
        assert store1["category_a"].to_list()[0] == 100.0
        assert store1["category_b"].to_list()[0] == 200.0
        
        # Check store_2
        store2 = result.filter(pl.col("store_id") == "store_2")
        assert store2["category_a"].to_list()[0] == 450.0
        assert store2["category_b"].to_list()[0] == 120.0
        
        # Check that missing values are filled with 0
        store3 = result.filter(pl.col("store_id") == "store_3")
        assert store3["category_a"].to_list()[0] == 0.0  # Missing value should be filled
        assert store3["category_b"].to_list()[0] == 250.0


class TestInternalNormalizedSalesData:
    """Tests for internal_normalized_sales_data asset."""
    
    def test_normalize_sales_data(self, mock_execution_context):
        """Test normalizing sales data."""
        # Create test data with pivoted sales
        pivoted_sales = pl.DataFrame({
            "store_id": ["store_1", "store_2", "store_3"],
            "category_a": [100.0, 450.0, 0.0],
            "category_b": [200.0, 120.0, 250.0],
        })
        
        # Configure normalization method in context
        mock_execution_context.resources.config.normalize_method = "min_max"
        
        # Execute the asset
        result = internal_normalized_sales_data(mock_execution_context, pivoted_sales)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "category_a" in result.columns
        assert "category_b" in result.columns
        
        # Verify normalization (values should be between 0 and 1)
        for col in ["category_a", "category_b"]:
            assert result[col].min() >= 0.0
            assert result[col].max() <= 1.0
        
        # For min_max normalization:
        # category_a: [100, 450, 0] -> [0.22, 1.0, 0.0]
        # category_b: [200, 120, 250] -> [0.8, 0.48, 1.0]
        # Check a few values with tolerance for floating point
        store1 = result.filter(pl.col("store_id") == "store_1")
        assert abs(store1["category_a"].to_list()[0] - 0.22) < 0.01
        assert abs(store1["category_b"].to_list()[0] - 0.8) < 0.01
        
        store3 = result.filter(pl.col("store_id") == "store_3")
        assert store3["category_a"].to_list()[0] == 0.0
        assert abs(store3["category_b"].to_list()[0] - 1.0) < 0.001 