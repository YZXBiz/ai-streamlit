"""Unit tests for SQL module functions and queries."""

import polars as pl

from clustering.core import sql


def test_execute_duckdb_query_polars_output():
    """Test the execute_duckdb_query function with polars output."""
    # Create test data
    test_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Simple query
    query = "SELECT * FROM test_data WHERE col1 > 1"

    # Execute query
    result = sql.execute_duckdb_query(query=query, dataframes={"test_data": test_df})

    # Check results
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2
    assert result["col1"].to_list() == [2, 3]
    assert result["col2"].to_list() == ["b", "c"]


def test_execute_duckdb_query_pandas_output():
    """Test the execute_duckdb_query function with pandas output."""
    # Create test data
    test_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Simple query
    query = "SELECT * FROM test_data WHERE col1 > 1"

    # Execute query
    result = sql.execute_duckdb_query(
        query=query, dataframes={"test_data": test_df}, output_format="pandas"
    )

    # Check results
    assert not isinstance(result, pl.DataFrame)
    assert len(result) == 2
    assert result["col1"].tolist() == [2, 3]
    assert result["col2"].tolist() == ["b", "c"]


def test_clean_need_state_query():
    """Test the clean need state query."""
    # Create test data with nulls
    test_df = pl.DataFrame(
        {
            "PRODUCT_ID": [1, 2, 3],
            "ATTRIBUTE_1": [None, "Value", "Test"],
            "ATTRIBUTE_2": ["A", None, "B"],
            "NEW_ITEM": [True, False, True],
            "TO_BE_DROPPED": [False, False, True],
        }
    )

    # Format and execute query
    query = sql.CLEAN_NEED_STATE_QUERY.format(table_name="test_data")
    result = sql.execute_duckdb_query(query=query, dataframes={"test_data": test_df})

    # Check results
    assert len(result) == 3
    # Check null handling
    assert result["ATTRIBUTE_1"][0] == "UNKNOWN"
    assert result["ATTRIBUTE_1"][1] == "Value"
    assert result["ATTRIBUTE_2"][1] == "UNKNOWN"
    # Check boolean casting
    assert isinstance(result["NEW_ITEM"][0], bool)
    assert isinstance(result["TO_BE_DROPPED"][0], bool)


def test_merge_sales_ns_query():
    """Test the merge sales and need state query."""
    # Create test sales data
    sales_df = pl.DataFrame(
        {"SKU_NBR": [1, 2, 3], "STORE_NBR": [101, 102, 103], "TOTAL_SALES": [1000, 2000, 3000]}
    )

    # Create test need state data
    ns_df = pl.DataFrame(
        {
            "PRODUCT_ID": [1, 2, 4],  # Note: 4 won't match
            "NEED_STATE": ["State1", "State2", "State4"],
        }
    )

    # Format and execute query
    query = sql.MERGE_SALES_NS_QUERY.format(
        sales_table="sales", ns_table="need_state", sku_col="SKU_NBR", product_col="PRODUCT_ID"
    )

    result = sql.execute_duckdb_query(
        query=query, dataframes={"sales": sales_df, "need_state": ns_df}
    )

    # Check results
    assert len(result) == 2  # Only 2 rows should match
    assert "NEED_STATE" in result.columns
    assert set(result["SKU_NBR"].to_list()) == {1, 2}  # Only SKUs 1 and 2 should match
    assert "State1" in result["NEED_STATE"].to_list()
    assert "State2" in result["NEED_STATE"].to_list()


def test_distribute_sales_query():
    """Test the distribute sales query."""
    # Create test data with duplicate SKU/store combinations
    test_df = pl.DataFrame(
        {
            "SKU_NBR": [1, 1, 2, 2],
            "STORE_NBR": [101, 101, 102, 102],
            "NEED_STATE": ["State1", "State1", "State2", "State2"],
            "CAT_DSC": ["Cat1", "Cat1", "Cat2", "Cat2"],
            "TOTAL_SALES": [100, 200, 300, 400],
        }
    )

    # Format and execute query
    query = sql.DISTRIBUTE_SALES_QUERY.format(table_name="test_data")
    result = sql.execute_duckdb_query(query=query, dataframes={"test_data": test_df})

    # Check results
    assert len(result) == 2  # Should be grouped into 2 rows
    assert result.filter(pl.col("SKU_NBR") == 1)["TOTAL_SALES"][0] == 300  # 100 + 200
    assert result.filter(pl.col("SKU_NBR") == 2)["TOTAL_SALES"][0] == 700  # 300 + 400
