"""Tests for Snowflake reader."""

import io
import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import duckdb
import pytest
import polars as pl
import pandas as pd
import snowflake.connector

from clustering.shared.io.readers.snowflake_reader import SnowflakeReader


class TestSnowflakeReader:
    """Test suite for SnowflakeReader."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.test_query = "SELECT * FROM test_table"
        self.mock_creds = {
            "SF_USER_NAME": "test_user",
            "SF_ACCOUNT": "test_account",
            "SF_DB": "test_db",
            "SF_WAREHOUSE": "test_warehouse",
            "SF_USER_ROLE": "test_role",
            "SF_INSECURE_MODE": "False",
        }
        self.mock_pkb = b"mock_private_key"
        self.mock_data = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

    @pytest.fixture
    def mock_tmp_cache_file(self, tmpdir):
        """Create a temporary cache file for testing."""
        cache_file = tmpdir.join("test_cache.duckdb")
        return str(cache_file)

    @patch("snowflake.connector.connect")
    def test_create_connection(self, mock_connect):
        """Test creating a Snowflake connection."""
        # Setup mocks for file operations
        mock_creds_content = json.dumps(self.mock_creds)
        
        # Create a context manager that mocks both file operations
        with patch("builtins.open", mock_open()) as m:
            # Set up the mock to return different content for different files
            # First call is for pkb file, second for creds file
            file_handle = m.return_value.__enter__.return_value
            
            # For the first call to open (pickle file), we'll handle the pickle.load separately
            # For the second call to open (creds file), return our mock credentials
            file_handle.read.return_value = mock_creds_content
            
            # Mock pickle.load to return our mock key
            with patch("pickle.load", return_value=self.mock_pkb):
                # Create the reader and test connection creation
                reader = SnowflakeReader(query=self.test_query)
                reader._create_connection()
                
                # Verify connection was created with correct parameters
                mock_connect.assert_called_once_with(
                    user=self.mock_creds["SF_USER_NAME"],
                    private_key=self.mock_pkb,
                    account=self.mock_creds["SF_ACCOUNT"],
                    database=self.mock_creds["SF_DB"],
                    warehouse=self.mock_creds["SF_WAREHOUSE"],
                    role=self.mock_creds["SF_USER_ROLE"],
                    insecure_mode=False,
                )

    @patch("os.path.exists")
    @patch("duckdb.connect")
    def test_load_cache_exists(self, mock_duckdb_connect, mock_exists, mock_tmp_cache_file):
        """Test loading cache when it exists and contains the query."""
        # Setup mocks
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        
        # Setup mock query result - a binary parquet file that contains the test data
        buffer = io.BytesIO()
        self.mock_data.write_parquet(buffer)
        buffer.seek(0)
        mock_conn.execute.return_value.fetchone.return_value = (buffer.getvalue(),)
        
        # Create the reader and test cache loading
        reader = SnowflakeReader(query=self.test_query, cache_file=mock_tmp_cache_file)
        
        # Mock the polars read_parquet function to return our test data
        with patch("polars.read_parquet", return_value=self.mock_data):
            result = reader._load_cache()
            
            # Verify cache was queried correctly
            mock_conn.execute.assert_called_once_with(
                "SELECT data FROM cache WHERE query = ?", 
                (self.test_query,)
            )
            
            # Verify cache result was returned
            assert isinstance(result, pl.DataFrame)
            assert result is self.mock_data  # Because we're mocking read_parquet

    @patch("os.path.exists")
    @patch("duckdb.connect")
    def test_load_cache_not_exists(self, mock_duckdb_connect, mock_exists, mock_tmp_cache_file):
        """Test loading cache when cache file doesn't exist."""
        # Setup mocks
        mock_exists.return_value = False
        
        # Create the reader and test cache loading
        reader = SnowflakeReader(query=self.test_query, cache_file=mock_tmp_cache_file)
        result = reader._load_cache()
        
        # Verify duckdb.connect was not called and None was returned
        mock_duckdb_connect.assert_not_called()
        assert result is None

    @patch("os.path.exists")
    @patch("duckdb.connect")
    def test_load_cache_catalog_exception(self, mock_duckdb_connect, mock_exists, mock_tmp_cache_file):
        """Test loading cache when a DuckDB CatalogException occurs."""
        # Setup mocks
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        
        # Make execute raise a CatalogException
        mock_conn.execute.side_effect = duckdb.CatalogException("Test exception")
        
        # Create the reader and test cache loading
        reader = SnowflakeReader(query=self.test_query, cache_file=mock_tmp_cache_file)
        
        # Capture printed output to check exception handling
        with patch("builtins.print") as mock_print:
            result = reader._load_cache()
            
            # Verify exception was caught and handled correctly
            mock_print.assert_called_once()
            assert "DuckDB CatalogException:" in mock_print.call_args[0][0]
            assert result is None

    @patch("os.makedirs")
    @patch("duckdb.connect")
    def test_save_cache(self, mock_duckdb_connect, mock_makedirs, mock_tmp_cache_file):
        """Test saving data to cache."""
        # Setup mocks
        mock_conn = MagicMock()
        mock_duckdb_connect.return_value = mock_conn
        
        # Create the reader and test cache saving
        reader = SnowflakeReader(query=self.test_query, cache_file=mock_tmp_cache_file)
        reader._save_cache(self.mock_data)
        
        # Verify directory was created
        mock_makedirs.assert_called_once_with(os.path.dirname(mock_tmp_cache_file), exist_ok=True)
        
        # Verify table was created
        mock_conn.execute.assert_any_call("CREATE TABLE IF NOT EXISTS cache (query VARCHAR, data BLOB)")
        
        # Verify data was inserted - we can't check the exact binary data, but we can check the query
        assert mock_conn.execute.call_args_list[1][0][0] == "INSERT INTO cache VALUES (?, ?)"
        assert mock_conn.execute.call_args_list[1][0][1][0] == self.test_query

    @patch.object(SnowflakeReader, "_create_connection")
    @patch.object(SnowflakeReader, "_load_cache")
    @patch.object(SnowflakeReader, "_save_cache")
    @patch("polars.read_database")
    def test_read_from_source_no_cache(self, mock_read_database, mock_save_cache, 
                                     mock_load_cache, mock_create_connection):
        """Test reading from Snowflake with cache disabled."""
        # Setup mocks
        mock_conn = MagicMock()
        mock_create_connection.return_value = mock_conn
        mock_read_database.return_value = self.mock_data
        
        # Create the reader with cache disabled
        reader = SnowflakeReader(query=self.test_query, use_cache=False)
        result = reader._read_from_source()
        
        # Verify cache was not checked
        mock_load_cache.assert_not_called()
        
        # Verify connection was created and used
        mock_create_connection.assert_called_once()
        mock_read_database.assert_called_once_with(query=self.test_query, connection=mock_conn)
        
        # Verify connection was closed
        mock_conn.close.assert_called_once()
        
        # Verify cache was not saved
        mock_save_cache.assert_not_called()
        
        # Verify correct data was returned
        assert result is self.mock_data

    @patch.object(SnowflakeReader, "_create_connection")
    @patch.object(SnowflakeReader, "_load_cache")
    @patch.object(SnowflakeReader, "_save_cache")
    @patch("polars.read_database")
    def test_read_from_source_with_cache_miss(self, mock_read_database, mock_save_cache, 
                                            mock_load_cache, mock_create_connection):
        """Test reading from Snowflake with cache enabled but cache miss."""
        # Setup mocks
        mock_conn = MagicMock()
        mock_create_connection.return_value = mock_conn
        mock_load_cache.return_value = None  # Cache miss
        mock_read_database.return_value = self.mock_data
        
        # Create the reader with cache enabled
        reader = SnowflakeReader(query=self.test_query, use_cache=True)
        result = reader._read_from_source()
        
        # Verify cache was checked
        mock_load_cache.assert_called_once()
        
        # Verify connection was created and used
        mock_create_connection.assert_called_once()
        mock_read_database.assert_called_once_with(query=self.test_query, connection=mock_conn)
        
        # Verify connection was closed
        mock_conn.close.assert_called_once()
        
        # Verify cache was saved
        mock_save_cache.assert_called_once_with(self.mock_data)
        
        # Verify correct data was returned
        assert result is self.mock_data

    @patch.object(SnowflakeReader, "_load_cache")
    @patch.object(SnowflakeReader, "_create_connection")
    def test_read_from_source_with_cache_hit(self, mock_create_connection, mock_load_cache):
        """Test reading from Snowflake with cache enabled and cache hit."""
        # Setup mocks
        mock_load_cache.return_value = self.mock_data  # Cache hit
        
        # Create the reader with cache enabled
        reader = SnowflakeReader(query=self.test_query, use_cache=True)
        result = reader._read_from_source()
        
        # Verify cache was checked
        mock_load_cache.assert_called_once()
        
        # Verify connection was not created or used
        mock_create_connection.assert_not_called()
        
        # Verify correct data was returned from cache
        assert result is self.mock_data 