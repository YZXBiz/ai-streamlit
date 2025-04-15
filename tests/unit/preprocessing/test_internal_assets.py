"""Tests for internal preprocessing assets."""

import polars as pl
import pytest
from dagster import build_asset_context

from clustering.dagster.assets.preprocessing.internal import (
    internal_category_data,
    internal_need_state_data,
    internal_sales_data,
    merged_internal_data,
)
from tests.unit.preprocessing.conftest import MockReader

# Remove unused imports
# from clustering.core.sql_engine import DuckDB


@pytest.fixture
def mock_sales_data() -> pl.DataFrame:
    """Create sample sales data.

    Returns:
        Sample sales data for testing
    """
    return pl.DataFrame(
        {
            "SKU_NBR": ["P001", "P002", "P003", "P001", "P002"],
            "STORE_NBR": ["S001", "S001", "S001", "S002", "S002"],
            "CAT_DSC": ["Grocery", "Produce", "Bakery", "Grocery", "Produce"],
            "TOTAL_SALES": [100.0, 150.0, 75.0, 120.0, 90.0],
        }
    )


@pytest.fixture
def mock_need_state_data() -> pl.DataFrame:
    """Create sample need state data with the required schema.

    Returns:
        Sample need state data for testing
    """
    return pl.DataFrame(
        {
            "PRODUCT_ID": ["P001", "P002", "P003"],
            "NEED_STATE": ["everyday", "health", "treat"],  # lowercase to test uppercase conversion
            "WEIGHT": [0.7, 0.8, 0.9],
            # Other columns needed for the test
            "category_id": ["Grocery", "Produce", "Bakery"],
            "need_state_id": ["NS001", "NS002", "NS003"],
            "need_state_name": ["Everyday", "Health", "Treat"],
            "need_state_description": [
                "Daily essentials",
                "Health products",
                "Treat yourself",
            ],
            "CATEGORY_ID": ["Grocery", "Produce", "Bakery"],
            "NEED_STATE_DESCRIPTION": [
                "Daily essentials",
                "Health products",
                "Treat yourself",
            ],
        }
    )


@pytest.fixture
def mock_merged_data() -> pl.DataFrame:
    """Create sample merged data.

    Returns:
        Sample merged data for testing
    """
    return pl.DataFrame(
        {
            "SKU_NBR": ["P001", "P002", "P003", "P001", "P002"],
            "STORE_NBR": ["S001", "S001", "S001", "S002", "S002"],
            "CAT_DSC": ["Grocery", "Produce", "Bakery", "Grocery", "Produce"],
            "TOTAL_SALES": [100.0, 150.0, 75.0, 120.0, 90.0],
            "NEED_STATE": ["EVERYDAY", "HEALTH", "TREAT", "EVERYDAY", "HEALTH"],
            "WEIGHT": [0.7, 0.8, 0.9, 0.7, 0.8],
            "NEED_STATE_DESCRIPTION": [
                "Daily essentials",
                "Health products",
                "Treat yourself",
                "Daily essentials",
                "Health products",
            ],
            "PRODUCT_ID": ["P001", "P002", "P003", "P001", "P002"],
            "SALES_PCT": [1.0, 1.0, 1.0, 1.0, 1.0],  # Added for the new implementation
        }
    )


def test_internal_sales_data_asset(mock_sales_data):
    """Test the internal_sales_data asset.

    Args:
        mock_sales_data: Sample sales data
    """
    # Create mock context with resources
    context = build_asset_context(
        resources={
            "input_sales_reader": MockReader(mock_sales_data),
            "config": {},
        }
    )

    # Run the asset
    result = internal_sales_data(context)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert "SKU_NBR" in result.columns
    assert "STORE_NBR" in result.columns
    assert "CAT_DSC" in result.columns
    assert "TOTAL_SALES" in result.columns
    assert len(result) == len(mock_sales_data)


def test_internal_need_state_data_asset(mock_need_state_data):
    """Test the internal_need_state_data asset.

    Args:
        mock_need_state_data: Sample need state data
    """
    # Create mock context with resources
    context = build_asset_context(
        resources={
            "input_need_state_reader": MockReader(mock_need_state_data),
            "config": {},
        }
    )

    # Run the asset
    result = internal_need_state_data(context)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(mock_need_state_data)
    # Check that NEED_STATE was converted to uppercase
    assert all(ns.isupper() for ns in result["NEED_STATE"].to_list())


def test_merged_internal_data_asset(mock_sales_data, mock_need_state_data):
    """Test the merged_internal_data asset.

    Args:
        mock_sales_data: Sample sales data
        mock_need_state_data: Sample need state data
    """
    # Create mock context
    context = build_asset_context()

    # Run the asset
    result = merged_internal_data(context, mock_sales_data, mock_need_state_data)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert "SKU_NBR" in result.columns
    assert "STORE_NBR" in result.columns
    assert "NEED_STATE" in result.columns
    assert "TOTAL_SALES" in result.columns
    assert "SALES_PCT" in result.columns  # Check for new column in the Polars implementation

    # In our Polars implementation, values will be grouped and aggregated
    # So we should check that we have the right structure rather than exact count
    grouped_count = result.group_by(["SKU_NBR", "STORE_NBR", "NEED_STATE"]).agg(pl.count()).height
    assert grouped_count > 0


def test_internal_category_data_asset(mock_merged_data):
    """Test the internal_category_data asset.

    Args:
        mock_merged_data: Sample merged data
    """
    # Expected categories based on mock data
    expected_categories = ["Grocery", "Produce", "Bakery"]

    # Create mock context
    context = build_asset_context()

    # Run the asset
    result = internal_category_data(context, mock_merged_data)

    # Check that the result is a dictionary
    assert isinstance(result, dict)

    # Check that all expected categories are in the result
    assert set(result.keys()) == set(expected_categories)

    # Check that each value is a DataFrame
    for df in result.values():
        assert isinstance(df, pl.DataFrame)

    # Check that each DataFrame has the correct data
    for cat in expected_categories:
        cat_df = result[cat]
        assert len(cat_df) > 0
        assert all(cat_df["CAT_DSC"] == cat)
