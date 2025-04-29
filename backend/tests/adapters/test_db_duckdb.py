"""Tests for the DuckDB adapter."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.adapters.db_duckdb import DuckDBDataSource


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


@pytest.fixture
def temp_parquet_file():
    """Create a temporary Parquet file for testing."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


class TestDuckDBDataSource:
    """Test the DuckDBDataSource adapter."""

    @pytest.mark.adapter
    def test_init_with_file_path(self, temp_csv_file):
        """Test initialization with a file path."""
        db = DuckDBDataSource(file_path=temp_csv_file)
        assert db.file_path == temp_csv_file
        assert db.connection is not None

    @pytest.mark.adapter
    def test_init_with_dataframe(self):
        """Test initialization with a DataFrame."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}
        )

        db = DuckDBDataSource(df=df, table_name="test_table")
        assert db.df is df
        assert db.table_name == "test_table"
        assert db.connection is not None

    @pytest.mark.adapter
    def test_get_df_from_file(self, temp_csv_file):
        """Test getting a DataFrame from a file."""
        db = DuckDBDataSource(file_path=temp_csv_file)
        df = db.get_df()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["id", "name", "value"]
        assert df["name"].iloc[0] == "Alice"

    @pytest.mark.adapter
    def test_get_df_from_dataframe(self):
        """Test getting a DataFrame from a DataFrame."""
        original_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}
        )

        db = DuckDBDataSource(df=original_df, table_name="test_table")
        df = db.get_df()

        assert isinstance(df, pd.DataFrame)
        assert df.equals(original_df)

    @pytest.mark.adapter
    def test_get_raw_data_csv(self, temp_csv_file):
        """Test getting raw data from a CSV file."""
        db = DuckDBDataSource(file_path=temp_csv_file)
        data = db.get_raw_data()

        assert isinstance(data, str)
        assert "id,name,value" in data
        assert "1,Alice,10.5" in data

    @pytest.mark.adapter
    def test_get_raw_data_dataframe(self):
        """Test getting raw data from a DataFrame."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}
        )

        db = DuckDBDataSource(df=df, table_name="test_table")
        data = db.get_raw_data()

        assert isinstance(data, str)
        assert "id,name,value" in data
        assert "1,Alice,10.5" in data

    @pytest.mark.adapter
    def test_get_metadata(self, temp_csv_file):
        """Test getting metadata."""
        db = DuckDBDataSource(file_path=temp_csv_file)
        metadata = db.get_metadata()

        assert isinstance(metadata, dict)
        assert "file_path" in metadata
        assert metadata["file_path"] == temp_csv_file
        assert "file_size" in metadata
        assert metadata["file_size"] > 0
        assert "table_name" in metadata

    @pytest.mark.adapter
    def test_get_schema(self, temp_csv_file):
        """Test getting schema."""
        db = DuckDBDataSource(file_path=temp_csv_file)
        schema = db.get_schema()

        assert isinstance(schema, dict)
        assert "columns" in schema
        columns = schema["columns"]

        assert len(columns) == 3
        # Check for id column (could be INTEGER or BIGINT depending on DuckDB version)
        id_col = next(col for col in columns if col["name"] == "id")
        assert id_col["type"] in ["INTEGER", "BIGINT"]

        assert {"name": "name", "type": "VARCHAR"} in columns
        assert {"name": "value", "type": "DOUBLE"} in columns
