"""Tests for the SQL source module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.data_source.sql_source import SQLSource


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
        }
    )


def test_init():
    """Test initializing a SQLSource."""
    # Create a SQLSource
    source = SQLSource(
        "sqlite:///test.db", "SELECT * FROM test_table", "test_source", "Test description"
    )

    # Check that it was initialized correctly
    assert source.name == "test_source"
    assert source.description == "Test description"
    assert source.connection_string == "sqlite:///test.db"
    assert source.query == "SELECT * FROM test_table"
    assert source.dialect == "sqlite"


def test_init_with_auto_description():
    """Test initializing a SQLSource with automatic description."""
    # Create a SQLSource without a description
    source = SQLSource("sqlite:///test.db", "SELECT * FROM test_table", "test_source")

    # Check that a description was generated
    assert source.description is not None
    assert "sqlite" in source.description
    assert "SELECT * FROM test_table" in source.description


def test_init_with_auto_dialect():
    """Test initializing a SQLSource with automatic dialect detection."""
    # Create a SQLSource without specifying a dialect
    source = SQLSource(
        "postgresql://user:pass@localhost/db", "SELECT * FROM test_table", "test_source"
    )

    # Check that the dialect was detected correctly
    assert source.dialect == "postgresql"


@patch("backend.data_source.sql_source.sqlalchemy.create_engine")
@patch("backend.data_source.sql_source.pd.read_sql")
@patch("backend.data_source.sql_source.pai.DataFrame")
def test_load(mock_dataframe, mock_read_sql, mock_create_engine, sample_dataframe):
    """Test loading data from a SQL database."""
    # Set up mocks
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    mock_read_sql.return_value = sample_dataframe
    mock_pai_df = MagicMock()
    mock_dataframe.return_value = mock_pai_df

    # Create a SQLSource
    source = SQLSource(
        "sqlite:///test.db", "SELECT * FROM test_table", "test_source", "Test description"
    )

    # Load the data
    result = source.load()

    # Check that the engine was created correctly
    mock_create_engine.assert_called_once_with("sqlite:///test.db")

    # Check that the query was executed correctly
    mock_read_sql.assert_called_once_with("SELECT * FROM test_table", mock_connection)

    # Check that the DataFrame was created correctly
    mock_dataframe.assert_called_once_with(
        sample_dataframe, name="test_source", description="Test description"
    )

    # Check that the engine was disposed
    mock_engine.dispose.assert_called_once()

    # Check that the result is the PandasAI DataFrame
    assert result is mock_pai_df


def test_get_source_info():
    """Test getting metadata about a SQL data source."""
    # Create a SQLSource
    source = SQLSource(
        "postgresql://user:password@localhost/db",
        "SELECT * FROM test_table",
        "test_source",
        "Test description",
    )

    # Get the source info
    info = source.get_source_info()

    # Check that it contains the expected information
    assert info["type"] == "sql"
    assert info["dialect"] == "postgresql"
    assert info["name"] == "test_source"
    assert info["description"] == "Test description"
    assert info["query"] == "SELECT * FROM test_table"

    # Check that the connection string is sanitized
    assert "password" not in info["connection"]
    assert "***" in info["connection"]


def test_sanitize_connection_string():
    """Test sanitizing a connection string."""
    # Create a SQLSource
    source = SQLSource(
        "postgresql://user:password@localhost/db", "SELECT * FROM test_table", "test_source"
    )

    # Sanitize the connection string
    sanitized = source._sanitize_connection_string(source.connection_string)

    # Check that the password is removed
    assert "password" not in sanitized
    assert "***" in sanitized
    assert "user" in sanitized
    assert "localhost" in sanitized
