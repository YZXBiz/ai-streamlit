"""
Tests for the DataFrameCollection class.

This module contains tests for the DataFrameCollection class, which is used
for organizing multiple dataframes into a logical collection.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.dataframe.collection import DataFrameCollection


@pytest.fixture
def sample_dataframes():
    """Create sample dataframes for testing."""
    # Create two sample dataframes
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "country": ["USA", "UK", "France", "Germany", "Japan"],
            "salary": [50000, 60000, 70000, 80000, 90000],
        }
    )

    return [df1, df2]


@pytest.fixture
def sample_names():
    """Create sample dataframe names for testing."""
    return ["people", "countries"]


def test_init(sample_dataframes, sample_names):
    """Test initializing a DataFrameCollection."""
    # Create a collection with specific names
    collection = DataFrameCollection(
        sample_dataframes, "test_collection", sample_names, "Test description"
    )

    # Check that it was initialized correctly
    assert collection.name == "test_collection"
    assert collection.description == "Test description"
    assert len(collection.dataframes) == 2
    assert collection.dataframe_names == sample_names

    # Create a collection with default names
    collection2 = DataFrameCollection(
        sample_dataframes, "test_collection2", description="Another description"
    )

    # Check default naming
    assert collection2.dataframe_names == ["dataframe_0", "dataframe_1"]


def test_add_dataframe(sample_dataframes, sample_names):
    """Test adding a dataframe to a collection."""
    # Create a collection with one dataframe
    collection = DataFrameCollection([sample_dataframes[0]], "test_collection", [sample_names[0]])

    # Create a new dataframe to add
    new_df = pd.DataFrame(
        {"id": [6, 7, 8], "product": ["Apple", "Banana", "Cherry"], "price": [1.0, 0.5, 2.0]}
    )

    # Add the dataframe with a name
    result = collection.add_dataframe(new_df, "products")

    # Check that it was added correctly
    assert result is True
    assert len(collection.dataframes) == 2
    assert collection.dataframe_names == ["people", "products"]
    assert collection.dataframes[1].equals(new_df)

    # Add another dataframe without a name
    another_df = pd.DataFrame({"id": [9, 10], "value": [100, 200]})
    result = collection.add_dataframe(another_df)

    # Check default naming
    assert result is True
    assert len(collection.dataframes) == 3
    assert collection.dataframe_names == ["people", "products", "dataframe_2"]


def test_add_duplicate_name(sample_dataframes, sample_names):
    """Test adding a dataframe with a duplicate name."""
    # Create a collection
    collection = DataFrameCollection(sample_dataframes, "test_collection", sample_names)

    # Try to add a dataframe with a duplicate name
    new_df = pd.DataFrame({"id": [6, 7, 8], "value": [100, 200, 300]})
    result = collection.add_dataframe(new_df, "people")

    # Check that it wasn't added
    assert result is False
    assert len(collection.dataframes) == 2


def test_remove_dataframe(sample_dataframes, sample_names):
    """Test removing a dataframe from a collection."""
    # Create a collection
    collection = DataFrameCollection(sample_dataframes, "test_collection", sample_names)

    # Remove a dataframe
    result = collection.remove_dataframe("people")

    # Check that it was removed correctly
    assert result is True
    assert len(collection.dataframes) == 1
    assert collection.dataframe_names == ["countries"]

    # Check that removing a non-existent dataframe returns False
    assert collection.remove_dataframe("non_existent") is False


def test_get_dataframe(sample_dataframes, sample_names):
    """Test getting a dataframe by name."""
    # Create a collection
    collection = DataFrameCollection(sample_dataframes, "test_collection", sample_names)

    # Get a dataframe
    df = collection.get_dataframe("people")

    # Check that it's the correct dataframe
    assert df.equals(sample_dataframes[0])

    # Check that getting a non-existent dataframe returns None
    assert collection.get_dataframe("non_existent") is None


def test_get_dataframes(sample_dataframes, sample_names):
    """Test getting all dataframes."""
    # Create a collection
    collection = DataFrameCollection(sample_dataframes, "test_collection", sample_names)

    # Get all dataframes
    dfs = collection.get_dataframes()

    # Check that they're the correct dataframes
    assert len(dfs) == 2
    assert dfs[0].equals(sample_dataframes[0])
    assert dfs[1].equals(sample_dataframes[1])


def test_get_query_context(sample_dataframes, sample_names):
    """Test getting the query context."""
    # Create a collection
    collection = DataFrameCollection(
        sample_dataframes, "test_collection", sample_names, "Test description"
    )

    # Get the query context
    context = collection.get_query_context()

    # Check that it contains the expected information
    assert "test_collection" in context
    assert "Test description" in context
    assert "people" in context
    assert "countries" in context
