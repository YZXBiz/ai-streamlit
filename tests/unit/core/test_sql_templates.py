"""Unit tests for SQL template functions."""

import polars as pl
import pytest

from clustering.core.sql_engine import SQL, DuckDB
from clustering.core.sql_templates import (
    clean_need_state,
    distribute_sales,
    get_categories,
    get_category_data,
    merge_sales_with_need_state,
)


@pytest.fixture
def sample_need_state_df() -> pl.DataFrame:
    """Create a sample need state DataFrame for testing."""
    return pl.DataFrame(
        {
            "category_id": ["1", "2", "3"],
            "need_state_id": ["NS1", "NS2", "NS3"],
            "need_state_name": ["Need State 1", "Need State 2", "Need State 3"],
            "need_state_description": ["Desc 1", "Desc 2", "Desc 3"],
        }
    )


@pytest.fixture
def sample_sales_df() -> pl.DataFrame:
    """Create a sample sales DataFrame for testing."""
    return pl.DataFrame(
        {
            "SKU_NBR": ["P0001", "P0002", "P0003"],
            "STORE_NBR": ["S001", "S002", "S003"],
            "CAT_DSC": ["1", "2", "3"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "need_state_id": ["NS1", "NS2", "NS3"],
            "price": [10.0, 15.0, 20.0],
            "weight": [1.0, 2.0, 3.0],
            "is_promotional": [False, True, False],
            "is_seasonal": [False, False, True],
            "store_size": ["small", "medium", "large"],
            "region": ["North", "South", "East"],
            "sales_units": [100, 200, 300],
            "TOTAL_SALES": [1000.0, 2000.0, 3000.0],
        }
    )


@pytest.fixture
def sample_cleaned_need_state_df() -> pl.DataFrame:
    """Create a sample cleaned need state DataFrame for testing."""
    return pl.DataFrame(
        {
            "CATEGORY_ID": ["1", "2", "3"],
            "NEED_STATE_ID": ["NS1", "NS2", "NS3"],
            "NEED_STATE_NAME": ["Need State 1", "Need State 2", "Need State 3"],
            "NEED_STATE_DESCRIPTION": ["Desc 1", "Desc 2", "Desc 3"],
            "PRODUCT_ID": [1, 2, 3],
            "NEED_STATE": ["Need State 1", "Need State 2", "Need State 3"],
            "CATEGORY": ["Desc 1", "Desc 2", "Desc 3"],
            "CDT": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_1": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_2": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_3": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_4": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_5": ["Unknown", "Unknown", "Unknown"],
            "ATTRIBUTE_6": ["Unknown", "Unknown", "Unknown"],
            "PLANOGRAM_DSC": ["Need State 1", "Need State 2", "Need State 3"],
            "PLANOGRAM_NBR": ["NS1", "NS2", "NS3"],
            "NEW_ITEM": [False, False, False],
            "TO_BE_DROPPED": [False, False, False],
        }
    )


@pytest.fixture
def sample_merged_df() -> pl.DataFrame:
    """Create a sample merged DataFrame for testing."""
    return pl.DataFrame(
        {
            "SKU_NBR": ["P0001", "P0002", "P0003"],
            "STORE_NBR": ["S001", "S002", "S003"],
            "DATE": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "CAT_DSC": ["1", "2", "3"],
            "need_state_id": ["NS1", "NS2", "NS3"],
            "price": [10.0, 15.0, 20.0],
            "weight": [1.0, 2.0, 3.0],
            "is_promotional": [False, True, False],
            "is_seasonal": [False, False, True],
            "store_size": ["small", "medium", "large"],
            "region": ["North", "South", "East"],
            "sales_units": [100, 200, 300],
            "TOTAL_SALES": [1000.0, 2000.0, 3000.0],
            "NEED_STATE": ["Need State 1", "Need State 2", "Need State 3"],
            "NEED_STATE_DESCRIPTION": ["Desc 1", "Desc 2", "Desc 3"],
        }
    )


def test_clean_need_state(sample_need_state_df: pl.DataFrame) -> None:
    """Test that clean_need_state generates the correct SQL and executes successfully."""
    # Get the SQL object
    sql_obj = clean_need_state(sample_need_state_df)

    # Check that it's a SQL object
    assert isinstance(sql_obj, SQL)

    # Execute the SQL with DuckDB to verify it works
    db = DuckDB()
    try:
        result = db.query(sql_obj)

        # Check result has expected columns
        expected_columns = [
            "CATEGORY_ID",
            "NEED_STATE_ID",
            "NEED_STATE_NAME",
            "NEED_STATE_DESCRIPTION",
            "PRODUCT_ID",
            "NEED_STATE",
            "CATEGORY",
            "CDT",
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
            "PLANOGRAM_DSC",
            "PLANOGRAM_NBR",
            "NEW_ITEM",
            "TO_BE_DROPPED",
        ]
        assert all(col in result.columns for col in expected_columns)

        # Check result has same number of rows as input
        assert len(result) == len(sample_need_state_df)
    finally:
        db.close()


def test_clean_need_state_missing_columns() -> None:
    """Test that clean_need_state raises a ValueError when required columns are missing."""
    # Create a DataFrame missing required columns
    invalid_df = pl.DataFrame({"category_id": ["1", "2", "3"]})

    # Expect a ValueError when calling clean_need_state
    with pytest.raises(ValueError) as excinfo:
        clean_need_state(invalid_df)

    # Check that the error message mentions the missing columns
    error_msg = str(excinfo.value)
    assert "Missing required columns" in error_msg
    assert "need_state_id" in error_msg
    assert "need_state_name" in error_msg
    assert "need_state_description" in error_msg


def test_merge_sales_with_need_state(
    sample_sales_df: pl.DataFrame, sample_cleaned_need_state_df: pl.DataFrame
) -> None:
    """Test that merge_sales_with_need_state generates correct SQL and executes successfully."""
    # Get the SQL object
    sql_obj = merge_sales_with_need_state(sample_sales_df, sample_cleaned_need_state_df)

    # Check that it's a SQL object
    assert isinstance(sql_obj, SQL)

    # Execute the SQL with DuckDB to verify it works
    db = DuckDB()
    try:
        result = db.query(sql_obj)

        # Check that result contains expected columns from both DataFrames
        expected_columns = [
            "SKU_NBR",
            "STORE_NBR",
            "DATE",
            "CAT_DSC",
            "TOTAL_SALES",
            "NEED_STATE",
            "NEED_STATE_DESCRIPTION",
        ]
        assert all(col in result.columns for col in expected_columns)
    finally:
        db.close()


def test_distribute_sales(sample_merged_df: pl.DataFrame) -> None:
    """Test that distribute_sales generates correct SQL and executes successfully."""
    # Get the SQL object
    sql_obj = distribute_sales(sample_merged_df)

    # Check that it's a SQL object
    assert isinstance(sql_obj, SQL)

    # Execute the SQL with DuckDB to verify it works
    db = DuckDB()
    try:
        result = db.query(sql_obj)

        # Check that result contains expected columns
        expected_columns = ["STORE_NBR", "SKU_NBR", "NEED_STATE", "CAT_DSC", "TOTAL_SALES"]
        assert all(col in result.columns for col in expected_columns)
    finally:
        db.close()


def test_get_categories(sample_merged_df: pl.DataFrame) -> None:
    """Test that get_categories generates correct SQL and executes successfully."""
    # Get the SQL object
    sql_obj = get_categories(sample_merged_df)

    # Check that it's a SQL object
    assert isinstance(sql_obj, SQL)

    # Execute the SQL with DuckDB to verify it works
    db = DuckDB()
    try:
        result = db.query(sql_obj)

        # Check that result contains expected column
        assert "CAT_DSC" in result.columns

        # Check that result has distinct values
        assert len(result) <= len(sample_merged_df)
    finally:
        db.close()


def test_get_category_data(sample_merged_df: pl.DataFrame) -> None:
    """Test that get_category_data generates correct SQL and executes successfully."""
    # Get the SQL object
    sql_obj = get_category_data(sample_merged_df, "1")

    # Check that it's a SQL object
    assert isinstance(sql_obj, SQL)

    # Execute the SQL with DuckDB to verify it works
    db = DuckDB()
    try:
        result = db.query(sql_obj)

        # Check that result has same columns as input
        assert set(result.columns) == set(sample_merged_df.columns)

        # Check that result only contains rows for the specified category
        if len(result) > 0:  # Only check if there are matching rows
            assert all(row["CAT_DSC"] == "1" for row in result.iter_rows(named=True))
    finally:
        db.close()
