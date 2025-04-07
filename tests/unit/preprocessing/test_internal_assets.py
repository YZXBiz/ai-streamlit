"""Tests for internal preprocessing assets."""

import sys
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

# Add package directory to path if not already installed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clustering.dagster.assets.preprocessing.internal import (
    internal_category_data,
    internal_need_state_data,
    internal_sales_data,
    merged_internal_data,
)
from tests.unit.preprocessing.conftest import MockReader


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
            "NEED_STATE": ["Everyday", "Health", "Treat"],
            "WEIGHT": [0.7, 0.8, 0.9],
            # Required columns for validation
            "category_id": ["Grocery", "Produce", "Bakery"],
            "need_state_id": ["NS001", "NS002", "NS003"],
            "need_state_name": ["Everyday", "Health", "Treat"],
            "need_state_description": [
                "Daily essentials",
                "Health products",
                "Treat yourself",
            ],
            # Required for merge function
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
            "NEED_STATE": ["Everyday", "Health", "Treat", "Everyday", "Health"],
            "WEIGHT": [0.7, 0.8, 0.9, 0.7, 0.8],
            # Required for distribute_sales function
            "NEED_STATE_DESCRIPTION": [
                "Daily essentials",
                "Health products",
                "Treat yourself",
                "Daily essentials",
                "Health products",
            ],
        }
    )


def test_internal_sales_data_asset(mock_sales_data):
    """Test the internal_sales_data asset.

    Args:
        mock_sales_data: Sample sales data
    """
    # Create mock context with resources
    context = dg.build_op_context(
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


def test_internal_need_state_data_asset(mock_need_state_data, monkeypatch):
    """Test the internal_need_state_data asset.

    Args:
        mock_need_state_data: Sample need state data
        monkeypatch: Pytest monkeypatch fixture
    """

    # Mock the DuckDB query method to return the input dataframe
    def mock_query(self, query, output_format=None):
        return mock_need_state_data

    # Apply the monkey patch
    from clustering.core.sql_engine import DuckDB

    monkeypatch.setattr(DuckDB, "query", mock_query)

    # Create mock context with resources
    context = dg.build_op_context(
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


def test_merged_internal_data_asset(
    mock_sales_data, mock_need_state_data, mock_merged_data, monkeypatch
):
    """Test the merged_internal_data asset.

    Args:
        mock_sales_data: Sample sales data
        mock_need_state_data: Sample need state data
        mock_merged_data: Sample merged data
        monkeypatch: Pytest monkeypatch fixture
    """

    # Mock the DuckDB query method to return the mock merged data
    def mock_query(self, query, output_format=None):
        return mock_merged_data

    # Apply the monkey patch
    from clustering.core.sql_engine import DuckDB

    monkeypatch.setattr(DuckDB, "query", mock_query)

    # Create mock context
    context = dg.build_op_context()

    # Run the asset
    result = merged_internal_data(context, mock_sales_data, mock_need_state_data)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert "SKU_NBR" in result.columns
    assert "NEED_STATE" in result.columns
    assert "TOTAL_SALES" in result.columns
    assert len(result) == len(mock_merged_data)


def test_internal_category_data_asset(mock_merged_data, monkeypatch):
    """Test the internal_category_data asset.

    Args:
        mock_merged_data: Sample merged data
        monkeypatch: Pytest monkeypatch fixture
    """
    # Expected categories
    categories = ["Grocery", "Produce", "Bakery"]

    # Mock data for each category
    category_dfs = {
        "Grocery": pl.DataFrame(
            {
                "SKU_NBR": ["P001", "P001"],
                "STORE_NBR": ["S001", "S002"],
                "CAT_DSC": ["Grocery", "Grocery"],
                "TOTAL_SALES": [100.0, 120.0],
                "NEED_STATE": ["Everyday", "Everyday"],
                "WEIGHT": [0.7, 0.7],
                "NEED_STATE_DESCRIPTION": ["Daily essentials", "Daily essentials"],
            }
        ),
        "Produce": pl.DataFrame(
            {
                "SKU_NBR": ["P002", "P002"],
                "STORE_NBR": ["S001", "S002"],
                "CAT_DSC": ["Produce", "Produce"],
                "TOTAL_SALES": [150.0, 90.0],
                "NEED_STATE": ["Health", "Health"],
                "WEIGHT": [0.8, 0.8],
                "NEED_STATE_DESCRIPTION": ["Health products", "Health products"],
            }
        ),
        "Bakery": pl.DataFrame(
            {
                "SKU_NBR": ["P003"],
                "STORE_NBR": ["S001"],
                "CAT_DSC": ["Bakery"],
                "TOTAL_SALES": [75.0],
                "NEED_STATE": ["Treat"],
                "WEIGHT": [0.9],
                "NEED_STATE_DESCRIPTION": ["Treat yourself"],
            }
        ),
    }

    # Mock the DuckDB query method
    def mock_query(self, query, output_format=None):
        if output_format == "raw":
            # Return categories for get_categories query
            class MockResult:
                def fetchall(self):
                    return [(cat,) for cat in categories]

            return MockResult()
        else:
            # Return the appropriate category dataframe for get_category_data query
            # Handle SQL objects by checking their string representation
            query_str = str(query)
            for cat in categories:
                if cat in query_str:
                    return category_dfs[cat]
            return pl.DataFrame()

    # Apply the monkey patch
    from clustering.core.sql_engine import DuckDB

    monkeypatch.setattr(DuckDB, "query", mock_query)

    # Create mock context
    context = dg.build_op_context()

    # Run the asset
    result = internal_category_data(context, mock_merged_data)

    # Check the result
    assert isinstance(result, dict)
    assert len(result) == len(categories)
    for cat in categories:
        assert cat in result
        assert isinstance(result[cat], pl.DataFrame)
