"""Unit tests for internal preprocessing assets."""

import polars as pl
import pytest
from dagster import build_op_context

from clustering.dagster.assets.preprocessing.internal import internal_normalized_sales_data


@pytest.fixture
def sample_sales_with_categories() -> pl.DataFrame:
    """Create a sample DataFrame of sales with categories for testing.
    
    Returns:
        A sample DataFrame with sales and categories.
    """
    return pl.DataFrame({
        "SKU_NBR": [101, 101, 101, 102, 102, 103],
        "STORE_NBR": [1, 1, 1, 2, 2, 3],
        "CAT_DSC": ["Health", "Health", "Health", "Beauty", "Beauty", "Grocery"],
        "NEED_STATE": ["NS1", "NS2", "NS3", "NS1", "NS2", "NS1"],
        "TOTAL_SALES": [300.0, 300.0, 300.0, 400.0, 400.0, 500.0]
    })


def test_internal_normalized_sales_data(sample_sales_with_categories: pl.DataFrame) -> None:
    """Test the normalization of sales data.
    
    This should distribute sales evenly across need states that share the same SKU/STORE.
    
    Args:
        sample_sales_with_categories: Sample DataFrame with sales and categories.
    """
    # Create a context object
    context = build_op_context()
    
    # Call the asset function
    result = internal_normalized_sales_data(context, sample_sales_with_categories)
    
    # Verify the result shape
    assert result.shape == sample_sales_with_categories.shape
    
    # Check that the function evenly distributed sales:
    # For SKU 101, STORE 1: sales should be 100.0 for each need state (300/3)
    sku_101_rows = result.filter((pl.col("SKU_NBR") == 101) & (pl.col("STORE_NBR") == 1))
    assert sku_101_rows.shape[0] == 3
    for row_idx in range(3):
        assert sku_101_rows[row_idx, "TOTAL_SALES"] == 100.0
    
    # For SKU 102, STORE 2: sales should be 200.0 for each need state (400/2)
    sku_102_rows = result.filter((pl.col("SKU_NBR") == 102) & (pl.col("STORE_NBR") == 2))
    assert sku_102_rows.shape[0] == 2
    for row_idx in range(2):
        assert sku_102_rows[row_idx, "TOTAL_SALES"] == 200.0
    
    # For SKU 103, STORE 3: sales should remain 500.0 (only one need state)
    sku_103_rows = result.filter((pl.col("SKU_NBR") == 103) & (pl.col("STORE_NBR") == 3))
    assert sku_103_rows.shape[0] == 1
    assert sku_103_rows[0, "TOTAL_SALES"] == 500.0 