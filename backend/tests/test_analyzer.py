"""
Tests for the PandasAnalyzer class.

This module contains tests for the PandasAnalyzer class, which is the main
entry point for data analysis using pandas DataFrames.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.core.analyzer import PandasAnalyzer


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "country": ["United States", "United Kingdom", "France", "Germany", "Japan"],
            "population": [331000000, 67000000, 65000000, 83000000, 126000000],
            "gdp": [21400000, 2700000, 2600000, 3800000, 5000000],
        }
    )


@pytest.fixture
def analyzer():
    """Create a PandasAnalyzer instance for testing."""
    return PandasAnalyzer()


def test_load_dataframe(analyzer, sample_dataframe):
    """Test loading a pandas DataFrame into the analyzer."""
    # Load the dataframe
    df = analyzer.load_dataframe(sample_dataframe, "test_d", "Test description")

    # Check that it was loaded correctly
    assert "test_d" in analyzer.dataframes
    assert df.equals(sample_dataframe)

    # Check that it was registered with the manager
    assert "test_d" in analyzer.manager.dataframes


def test_get_dataframe(analyzer, sample_dataframe):
    """Test retrieving a dataframe by name."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d")

    # Retrieve it
    df = analyzer.get_dataframe("test_d")

    # Check that it's the correct dataframe
    assert df.equals(sample_dataframe)

    # Check that a non-existent dataframe returns None
    assert analyzer.get_dataframe("non_existent") is None


def test_get_dataframe_preview(analyzer, sample_dataframe):
    """Test getting a preview of a dataframe."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d")

    # Get a preview
    preview = analyzer.get_dataframe_preview("test_d", rows=2)

    # Check that it's the correct preview
    assert len(preview) == 2
    assert preview.equals(sample_dataframe.head(2))


def test_get_dataframe_schema(analyzer, sample_dataframe):
    """Test getting the schema of a dataframe."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d", "Test description")

    # Get the schema
    schema = analyzer.get_dataframe_schema("test_d")

    # Check that it contains the expected information
    assert schema["columns"] == ["country", "population", "gdp"]
    assert schema["shape"] == (5, 3)
    assert schema["name"] == "test_d"
    assert schema["description"] == "Test description"

    # Check dtypes
    assert "country" in schema["dtypes"]
    assert "population" in schema["dtypes"]
    assert "gdp" in schema["dtypes"]


def test_get_dataframe_stats(analyzer, sample_dataframe):
    """Test getting statistics for a dataframe."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d")

    # Get the statistics
    stats = analyzer.get_dataframe_stats("test_d")

    # Check that it contains the expected information
    assert "population" in stats.index
    assert "gdp" in stats.index
    assert "count" in stats.columns
    assert "mean" in stats.columns


def test_create_collection(analyzer, sample_dataframe):
    """Test creating a collection of dataframes."""
    # Load two dataframes
    analyzer.load_dataframe(sample_dataframe, "df1")
    analyzer.load_dataframe(sample_dataframe, "df2")

    # Create a collection
    collection = analyzer.create_collection(["df1", "df2"], "test_collection", "Test description")

    # Check that the collection was created correctly
    assert collection.name == "test_collection"
    assert collection.description == "Test description"
    assert len(collection.dataframes) == 2

    # Check that it was registered with the manager
    assert "test_collection" in analyzer.manager.collections


def test_clear_dataframes(analyzer, sample_dataframe):
    """Test clearing all dataframes."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d")

    # Clear all dataframes
    analyzer.clear_dataframes()

    # Check that all dataframes were cleared
    assert len(analyzer.dataframes) == 0
    assert len(analyzer.manager.dataframes) == 0


@pytest.mark.parametrize(
    "file_type,file_content",
    [
        ("csv", "country,population,gdp\nUSA,331000000,21400000\nUK,67000000,2700000"),
        ("parquet", None),  # We'll create a parquet file in the test
    ],
)
def test_load_file(analyzer, file_type, file_content, sample_dataframe):
    """Test loading data from a file."""
    with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as temp_file:
        try:
            # Create the file
            if file_type == "csv":
                temp_file.write(file_content.encode())
                temp_file.flush()
            elif file_type == "parquet":
                # For parquet, we need to use pandas to write the file
                sample_dataframe.to_parquet(temp_file.name)

            # Load the file
            if file_type == "csv":
                analyzer.load_csv(temp_file.name, "test_d", "Test description")
            elif file_type == "parquet":
                analyzer.load_parquet(temp_file.name, "test_d", "Test description")

            # Check that it was loaded correctly
            assert "test_d" in analyzer.dataframes

            # Check that it was registered with the manager
            assert "test_d" in analyzer.manager.dataframes
        finally:
            # Clean up
            os.unlink(temp_file.name)


def test_get_dataframe_names(analyzer, sample_dataframe):
    """Test getting all dataframe names."""
    # Load a dataframe
    analyzer.load_dataframe(sample_dataframe, "test_d")

    # Get all dataframe names
    names = analyzer.get_dataframe_names()

    # Check that the list contains the expected name
    assert names == ["test_d"]


def test_get_collection_names(analyzer):
    """Test getting all collection names."""
    # Initially, there should be no collections
    names = analyzer.get_collection_names()
    assert names == []
