"""Tests for the SnowflakeReader class."""

import io
import os
import pickle
from unittest import mock

import duckdb
import polars as pl
import pytest
import snowflake.connector

from clustering.shared.io.readers.snowflake_reader import SnowflakeReader


class TestSnowflakeReader:
    """Tests for SnowflakeReader."""

    def setup_method(self):
        """Set up before each test."""
        self.temp_cache_file = "test_snowflake_cache.duckdb"
        self.temp_pkb_file = "test_pkb.pkl"
        self.temp_creds_file = "test_sf_creds.json"

        # Clean up existing files
        for file in [self.temp_cache_file, self.temp_pkb_file, self.temp_creds_file]:
            if os.path.exists(file):
                os.remove(file)

    def teardown_method(self):
        """Clean up after each test."""
        for file in [self.temp_cache_file, self.temp_pkb_file, self.temp_creds_file]:
            if os.path.exists(file):
                os.remove(file)

    @mock.patch("snowflake.connector.connect")
    def test_create_connection(self, mock_connect):
        """Test creating a Snowflake connection."""
        # Create mock PKB file
        with open(self.temp_pkb_file, "wb") as f:
            pickle.dump("mock_private_key", f)

        # Create mock credentials file
        with open(self.temp_creds_file, "w") as f:
            f.write("""
            {
                "SF_USER_NAME": "test_user",
                "SF_ACCOUNT": "test_account",
                "SF_DB": "test_db",
                "SF_WAREHOUSE": "test_warehouse",
                "SF_USER_ROLE": "test_role",
                "SF_INSECURE_MODE": "True"
            }
            """)

        # Create reader with our mock files
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.pkb_path = self.temp_pkb_file
        reader.creds_path = self.temp_creds_file

        # Mock connection object
        mock_conn = mock.MagicMock()
        mock_connect.return_value = mock_conn

        # Call the method under test
        result = reader._create_connection()

        # Verify the connection was created with correct parameters
        mock_connect.assert_called_once()
        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["user"] == "test_user"
        assert call_kwargs["private_key"] == "mock_private_key"
        assert call_kwargs["account"] == "test_account"
        assert call_kwargs["database"] == "test_db"
        assert call_kwargs["warehouse"] == "test_warehouse"
        assert call_kwargs["role"] == "test_role"
        assert call_kwargs["insecure_mode"] is True

        assert result == mock_conn

    @mock.patch("duckdb.connect")
    def test_load_cache_with_results(self, mock_connect):
        """Test loading cached query results when they exist."""
        # Create a mock DuckDB connection
        mock_conn = mock.MagicMock()
        mock_connect.return_value = mock_conn

        # Set up the mock to return a result
        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_parquet_bytes = io.BytesIO()
        test_data.write_parquet(mock_parquet_bytes)
        mock_parquet_bytes.seek(0)
        bytes_data = mock_parquet_bytes.getvalue()

        mock_conn.execute().fetchone.return_value = (bytes_data,)

        # Create a mock file to make os.path.exists return True
        open(self.temp_cache_file, "a").close()

        # Create reader with our mock cache file
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.cache_file = self.temp_cache_file

        # Call the method under test
        with mock.patch("polars.read_parquet", return_value=test_data):
            result = reader._load_cache()

        # Verify the result
        assert result is test_data
        assert mock_connect.called
        mock_conn.execute.assert_called_with(
            "SELECT data FROM cache WHERE query = ?", ("SELECT * FROM test_table",)
        )

    @mock.patch("duckdb.connect")
    def test_load_cache_no_results(self, mock_connect):
        """Test loading cached query results when none exist."""
        # Create a mock DuckDB connection
        mock_conn = mock.MagicMock()
        mock_connect.return_value = mock_conn

        # Set up the mock to return None (no result)
        mock_cursor = mock.MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        # Create a mock file to make os.path.exists return True
        open(self.temp_cache_file, "a").close()

        # Create reader with our mock cache file
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.cache_file = self.temp_cache_file

        # Call the method under test
        result = reader._load_cache()

        # Verify the result
        assert result is None
        assert mock_connect.called
        mock_conn.execute.assert_called_with(
            "SELECT data FROM cache WHERE query = ?", ("SELECT * FROM test_table",)
        )
        mock_cursor.fetchone.assert_called_once()

    @mock.patch("duckdb.connect")
    def test_load_cache_no_cache_file(self, mock_connect):
        """Test loading cached query results when the cache file doesn't exist."""
        # Create reader with a nonexistent cache file
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.cache_file = self.temp_cache_file

        # Call the method under test
        result = reader._load_cache()

        # Verify the result
        assert result is None
        mock_connect.assert_not_called()

    @mock.patch("duckdb.connect")
    def test_load_cache_catalog_exception(self, mock_connect):
        """Test loading cached query results when a CatalogException occurs."""
        # Create a mock DuckDB connection
        mock_conn = mock.MagicMock()
        mock_connect.return_value = mock_conn

        # Set up the mock to raise a CatalogException
        mock_conn.execute.side_effect = duckdb.CatalogException("Table does not exist")

        # Create a mock file to make os.path.exists return True
        open(self.temp_cache_file, "a").close()

        # Create reader with our mock cache file
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.cache_file = self.temp_cache_file

        # Call the method under test
        with mock.patch("builtins.print") as mock_print:
            result = reader._load_cache()

        # Verify the result
        assert result is None
        assert mock_connect.called
        mock_conn.execute.assert_called_with(
            "SELECT data FROM cache WHERE query = ?", ("SELECT * FROM test_table",)
        )
        mock_print.assert_called()

    @mock.patch("duckdb.connect")
    @mock.patch("os.makedirs")
    def test_save_cache(self, mock_makedirs, mock_connect):
        """Test saving query results to cache."""
        # Create a mock DuckDB connection
        mock_conn = mock.MagicMock()
        mock_connect.return_value = mock_conn

        # Create test data
        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Create reader with our mock cache file
        reader = SnowflakeReader(query="SELECT * FROM test_table")
        reader.cache_file = self.temp_cache_file

        # Call the method under test
        reader._save_cache(test_data)

        # Verify that the directory was created
        mock_makedirs.assert_called_once()

        # Verify that the DuckDB connection was created
        mock_connect.assert_called_with(self.temp_cache_file)

        # Verify that the table was created
        mock_conn.execute.assert_any_call(
            "CREATE TABLE IF NOT EXISTS cache (query VARCHAR, data BLOB)"
        )

        # Verify that data was inserted
        assert mock_conn.execute.call_count == 2
        assert mock_conn.execute.call_args_list[1][0][0] == "INSERT INTO cache VALUES (?, ?)"
        assert mock_conn.execute.call_args_list[1][0][1][0] == "SELECT * FROM test_table"
        assert isinstance(mock_conn.execute.call_args_list[1][0][1][1], bytes)

    @mock.patch.object(SnowflakeReader, "_create_connection")
    @mock.patch.object(SnowflakeReader, "_load_cache")
    @mock.patch.object(SnowflakeReader, "_save_cache")
    def test_read_from_source_with_cache_hit(
        self, mock_save_cache, mock_load_cache, mock_create_connection
    ):
        """Test reading data from Snowflake when cache is hit."""
        # Create test data that will be "found" in the cache
        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_load_cache.return_value = test_data

        # Create reader
        reader = SnowflakeReader(query="SELECT * FROM test_table", use_cache=True)

        # Call the method under test
        result = reader._read_from_source()

        # Verify the result
        assert result is test_data
        mock_load_cache.assert_called_once()
        mock_create_connection.assert_not_called()
        mock_save_cache.assert_not_called()

    @mock.patch.object(SnowflakeReader, "_create_connection")
    @mock.patch.object(SnowflakeReader, "_load_cache")
    @mock.patch.object(SnowflakeReader, "_save_cache")
    @mock.patch("polars.read_database")
    def test_read_from_source_with_cache_miss(
        self, mock_read_database, mock_save_cache, mock_load_cache, mock_create_connection
    ):
        """Test reading data from Snowflake when cache is missed."""
        # No cached data
        mock_load_cache.return_value = None

        # Create mock connection
        mock_conn = mock.MagicMock()
        mock_create_connection.return_value = mock_conn

        # Create test data that will be "read" from the database
        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_read_database.return_value = test_data

        # Create reader
        reader = SnowflakeReader(query="SELECT * FROM test_table", use_cache=True)

        # Call the method under test
        result = reader._read_from_source()

        # Verify the result
        assert result is test_data
        mock_load_cache.assert_called_once()
        mock_create_connection.assert_called_once()
        mock_read_database.assert_called_with(
            query="SELECT * FROM test_table", connection=mock_conn
        )
        mock_save_cache.assert_called_with(test_data)
        mock_conn.close.assert_called_once()

    @mock.patch.object(SnowflakeReader, "_create_connection")
    @mock.patch.object(SnowflakeReader, "_load_cache")
    @mock.patch.object(SnowflakeReader, "_save_cache")
    @mock.patch("polars.read_database")
    def test_read_from_source_no_cache(
        self, mock_read_database, mock_save_cache, mock_load_cache, mock_create_connection
    ):
        """Test reading data from Snowflake with caching disabled."""
        # Create mock connection
        mock_conn = mock.MagicMock()
        mock_create_connection.return_value = mock_conn

        # Create test data that will be "read" from the database
        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_read_database.return_value = test_data

        # Create reader with caching disabled
        reader = SnowflakeReader(query="SELECT * FROM test_table", use_cache=False)

        # Call the method under test
        result = reader._read_from_source()

        # Verify the result
        assert result is test_data
        mock_load_cache.assert_not_called()
        mock_create_connection.assert_called_once()
        mock_read_database.assert_called_with(
            query="SELECT * FROM test_table", connection=mock_conn
        )
        mock_save_cache.assert_not_called()
        mock_conn.close.assert_called_once()
