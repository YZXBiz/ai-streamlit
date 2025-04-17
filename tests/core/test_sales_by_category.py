"""Unit tests for sales by category asset."""

import polars as pl
import pytest
from dagster import build_op_context

from clustering.dagster.assets.preprocessing.internal import internal_sales_by_category


@pytest.fixture
def sample_normalized_sales_data() -> pl.DataFrame:
    """Create a sample DataFrame of normalized sales data for testing.
    
    Returns:
        A sample DataFrame with normalized sales.
    """
    return pl.DataFrame({
        "SKU_NBR": [101, 101, 101, 102, 102, 103],
        "STORE_NBR": [1, 1, 1, 2, 2, 3],
        "CAT_DSC": ["Health", "Health", "Health", "Beauty", "Beauty", "Grocery"],
        "NEED_STATE": ["NS1", "NS2", "NS3", "NS1", "NS2", "NS1"],
        "TOTAL_SALES": [100.0, 100.0, 100.0, 200.0, 200.0, 500.0]
    })


def test_internal_sales_by_category(sample_normalized_sales_data: pl.DataFrame) -> None:
    """Test the conversion of sales data to category-specific DataFrames with percentages.
    
    Args:
        sample_normalized_sales_data: Sample DataFrame with normalized sales.
    """
    # Create a context object
    context = build_op_context()
    
    # Call the asset function
    result = internal_sales_by_category(context, sample_normalized_sales_data)
    
    # Verify the result is a dictionary
    assert isinstance(result, dict)
    
    # Verify that we have the expected categories
    assert set(result.keys()) == {"Health", "Beauty", "Grocery"}
    
    # Test the Health category
    health_df = result["Health"]
    assert isinstance(health_df, pl.DataFrame)
    assert "STORE_NBR" in health_df.columns
    
    # For store 1, there are 3 need states with equal sales (100 each)
    # So each should be 33.33% of the total (300)
    store_1_row = health_df.filter(pl.col("STORE_NBR") == 1)
    assert store_1_row.shape[0] == 1
    
    # The actual column names depend on need state naming in the pivoted output
    # Let's verify the percentages are correct regardless of column names
    need_state_columns = [col for col in health_df.columns if col != "STORE_NBR"]
    assert len(need_state_columns) == 3  # Should have 3 need state columns
    
    # Each need state should be ~33.33% of sales
    for col in need_state_columns:
        pct_value = store_1_row.select(col).item()
        assert abs(pct_value - 33.33) < 0.1  # Allow for small rounding errors
    
    # Test the Beauty category
    beauty_df = result["Beauty"]
    assert isinstance(beauty_df, pl.DataFrame)
    
    # For store 2, there are 2 need states with equal sales (200 each)
    # So each should be 50% of the total (400)
    store_2_row = beauty_df.filter(pl.col("STORE_NBR") == 2)
    assert store_2_row.shape[0] == 1
    
    # Verify the two need state columns have ~50% each
    beauty_need_state_columns = [col for col in beauty_df.columns if col != "STORE_NBR"]
    assert len(beauty_need_state_columns) == 2
    
    for col in beauty_need_state_columns:
        pct_value = store_2_row.select(col).item()
        assert abs(pct_value - 50.0) < 0.1
    
    # Test the Grocery category
    grocery_df = result["Grocery"]
    assert isinstance(grocery_df, pl.DataFrame)
    
    # For store 3, there is only 1 need state with 500 sales
    # So it should be 100% of the total
    store_3_row = grocery_df.filter(pl.col("STORE_NBR") == 3)
    assert store_3_row.shape[0] == 1
    
    # Verify the need state column has 100%
    grocery_need_state_column = [col for col in grocery_df.columns if col != "STORE_NBR"][0]
    pct_value = store_3_row.select(grocery_need_state_column).item()
    assert abs(pct_value - 100.0) < 0.1 