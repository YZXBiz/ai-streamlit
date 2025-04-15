"""Tests for Snowflake reader and writer with mocked connections."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import snowflake.connector

from clustering.io.readers.snowflake_reader import SnowflakeReader
from clustering.io.writers.snowflake_writer import SnowflakeWriter


@pytest.fixture
def mock_snowflake_connection():
    """Create a mock Snowflake connection."""
    mock_conn = MagicMock(spec=snowflake.connector.SnowflakeConnection)
    return mock_conn


@pytest.fixture
def mock_cursor():
    """Create a mock cursor with sample data."""
    mock_cursor = MagicMock()
    mock_cursor.description = [
        ("id", None, None, None, None, None, None),
        ("name", None, None, None, None, None, None),
        ("value", None, None, None, None, None, None),
    ]
    mock_cursor.fetchall.return_value = [(1, "Alice", 10.1), (2, "Bob", 20.2), (3, "Charlie", 30.3)]
    return mock_cursor


@pytest.fixture
def mock_duckdb_connection():
    """Create a mock DuckDB connection."""
    mock_conn = MagicMock()

    # Mock the execute method to handle the cache operations
    def mock_execute(*args, **kwargs):
        mock_conn.last_query = args[1] if len(args) > 1 else ""
        mock_conn.last_params = args[2] if len(args) > 2 else None
        mock_cursor = MagicMock()

        # Return None for SELECT to simulate cache miss
        if mock_conn.last_query.startswith("SELECT"):
            mock_cursor.fetchone.return_value = None

        return mock_cursor

    mock_conn.execute = mock_execute
    return mock_conn


@pytest.mark.parametrize("use_cache", [True, False])
def test_snowflake_reader_init(use_cache):
    """Test SnowflakeReader initialization."""
    reader = SnowflakeReader(
        query="SELECT * FROM test_table",
        use_cache=use_cache,
        cache_file="test_cache.duckdb",
        pkb_path="test_pkb.pkl",
        creds_path="test_creds.json",
    )

    assert reader.query == "SELECT * FROM test_table"
    assert reader.use_cache is use_cache
    assert reader.cache_file == "test_cache.duckdb"
    assert reader.pkb_path == "test_pkb.pkl"
    assert reader.creds_path == "test_creds.json"


@patch("clustering.io.readers.snowflake_reader.duckdb.connect")
@patch("clustering.io.readers.snowflake_reader.os.path.exists")
@patch("clustering.io.readers.snowflake_reader.SnowflakeReader._create_connection")
def test_snowflake_reader_read_with_cache(
    mock_create_conn, mock_exists, mock_duckdb_connect, sample_data, mock_snowflake_connection
):
    """Test SnowflakeReader read method with cache hit."""
    # Setup mocks
    mock_exists.return_value = True
    mock_duckdb_connect.return_value = MagicMock()

    # Create buffer with sample data in parquet format
    buffer = BytesIO()
    sample_data.write_parquet(buffer)
    buffer.seek(0)

    # Configure the duckdb mock to return cached data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (buffer.getvalue(),)
    mock_duckdb_connect.return_value.execute.return_value = mock_cursor

    # Create the reader and test cache hit
    reader = SnowflakeReader(query="SELECT * FROM test_table", use_cache=True)
    result = reader.read()

    # Verify cache was checked and connection was NOT created
    mock_exists.assert_called_once()
    mock_duckdb_connect.assert_called_once()
    mock_create_conn.assert_not_called()

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape


@patch("clustering.io.readers.snowflake_reader.pl.read_database")
@patch("clustering.io.readers.snowflake_reader.duckdb.connect")
@patch("clustering.io.readers.snowflake_reader.os.path.exists")
@patch("clustering.io.readers.snowflake_reader.SnowflakeReader._create_connection")
def test_snowflake_reader_read_without_cache(
    mock_create_conn,
    mock_exists,
    mock_duckdb_connect,
    mock_read_database,
    sample_data,
    mock_snowflake_connection,
):
    """Test SnowflakeReader read method with cache miss."""
    # Setup mocks
    mock_exists.return_value = True
    mock_create_conn.return_value = mock_snowflake_connection
    mock_duckdb_connect.return_value = MagicMock()

    # Configure duckdb mock to simulate cache miss
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_duckdb_connect.return_value.execute.return_value = mock_cursor

    # Configure the read_database mock
    mock_read_database.return_value = sample_data

    # Create the reader and test cache miss flow
    reader = SnowflakeReader(query="SELECT * FROM test_table", use_cache=True)
    result = reader.read()

    # Verify database was queried and cache was saved
    mock_exists.assert_called_once()
    mock_duckdb_connect.assert_called()
    mock_create_conn.assert_called_once()
    mock_read_database.assert_called_once()

    # Verify the query was saved to cache
    mock_duckdb_connect.return_value.execute.assert_called_with(
        "INSERT INTO cache VALUES (?, ?)",
        ("SELECT * FROM test_table", mock_read_database.return_value.write_parquet()),
    )

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape


def test_snowflake_writer_init():
    """Test SnowflakeWriter initialization."""
    writer = SnowflakeWriter(
        table="test_table",
        database="test_db",
        schema="test_schema",
        auto_create_table=True,
        overwrite=True,
        pkb_path="test_pkb.pkl",
        creds_path="test_creds.json",
    )

    assert writer.table == "test_table"
    assert writer.database == "test_db"
    assert writer.schema == "test_schema"
    assert writer.auto_create_table is True
    assert writer.overwrite is True
    assert writer.pkb_path == "test_pkb.pkl"
    assert writer.creds_path == "test_creds.json"


@patch("clustering.io.writers.snowflake_writer.snowflake.connector.pandas_tools.write_pandas")
@patch("clustering.io.writers.snowflake_writer.SnowflakeWriter._create_connection")
def test_snowflake_writer_write(
    mock_create_conn, mock_write_pandas, sample_data, mock_snowflake_connection
):
    """Test SnowflakeWriter write method."""
    # Setup mocks
    mock_create_conn.return_value = mock_snowflake_connection

    # Create the writer and test write method
    writer = SnowflakeWriter(
        table="test_table",
        database="test_db",
        schema="test_schema",
    )
    writer.write(sample_data)

    # Verify connection was created and write_pandas was called
    mock_create_conn.assert_called_once()
    mock_write_pandas.assert_called_once()

    # Check the write_pandas args
    args, kwargs = mock_write_pandas.call_args
    assert kwargs["table_name"] == "test_table"
    assert kwargs["database"] == "test_db"
    assert kwargs["schema"] == "test_schema"
    assert kwargs["auto_create_table"] is True
    assert kwargs["overwrite"] is True
