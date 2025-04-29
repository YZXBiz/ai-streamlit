"""Tests for the DuckDB implementation."""

from collections.abc import Generator

import pandas as pd
import pytest

from backend.app.adapters.db_duckdb import DuckDBManager


@pytest.fixture
def duckdb_manager() -> Generator[DuckDBManager, None, None]:
    """Create a test DuckDB manager."""
    # Create a manager with in-memory database
    manager = DuckDBManager()

    yield manager

    # Clean up
    manager.close()


def test_register_dataframe(duckdb_manager: DuckDBManager) -> None:
    """Test registering a pandas DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Check that the table was registered
    assert "test_table" in duckdb_manager.tables

    # Check that the table has the correct metadata
    assert duckdb_manager.tables["test_table"]["source"] == "pandas"
    assert duckdb_manager.tables["test_table"]["type"] == "dataframe"
    assert duckdb_manager.tables["test_table"]["row_count"] == 3


def test_execute_query(duckdb_manager: DuckDBManager) -> None:
    """Test executing a SQL query."""
    # Create a test DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Execute a query
    result = duckdb_manager.execute_query("SELECT * FROM test_table WHERE age > 25")

    # Check that the result is correct
    assert len(result) == 2
    assert result["name"].tolist() == ["Bob", "Charlie"]
    assert result["age"].tolist() == [30, 35]


def test_to_pandas(duckdb_manager: DuckDBManager) -> None:
    """Test converting a DuckDB table to a pandas DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Convert to pandas
    result = duckdb_manager.to_pandas("test_table")

    # Check that the result is correct
    assert len(result) == 3
    assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
    assert result["age"].tolist() == [25, 30, 35]


def test_get_table_schema(duckdb_manager: DuckDBManager) -> None:
    """Test getting the schema of a table."""
    # Create a test DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Get the schema
    schema = duckdb_manager.get_table_schema("test_table")

    # Check that the schema is correct
    assert "id" in schema
    assert "name" in schema
    assert "age" in schema

    # Check that the types are correct (DuckDB types may vary by version)
    assert "INTEGER" in schema["id"].upper() or "INT" in schema["id"].upper()
    assert "VARCHAR" in schema["name"].upper() or "STRING" in schema["name"].upper()
    assert "INTEGER" in schema["age"].upper() or "INT" in schema["age"].upper()


def test_get_table_preview(duckdb_manager: DuckDBManager) -> None:
    """Test getting a preview of a table."""
    # Create a test DataFrame with more than 10 rows
    df = pd.DataFrame({"id": range(1, 21), "value": range(101, 121)})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Get a preview with default limit
    preview = duckdb_manager.get_table_preview("test_table")

    # Check that the preview has the correct number of rows
    assert len(preview) == 10

    # Get a preview with custom limit
    preview = duckdb_manager.get_table_preview("test_table", limit=5)

    # Check that the preview has the correct number of rows
    assert len(preview) == 5


def test_drop_table(duckdb_manager: DuckDBManager) -> None:
    """Test dropping a table."""
    # Create a test DataFrame
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    # Register the DataFrame
    duckdb_manager.register_dataframe(df, "test_table")

    # Check that the table exists
    assert "test_table" in duckdb_manager.tables

    # Drop the table
    duckdb_manager.drop_table("test_table")

    # Check that the table was dropped
    assert "test_table" not in duckdb_manager.tables

    # Check that dropping a non-existent table doesn't raise an error
    duckdb_manager.drop_table("non_existent_table")
