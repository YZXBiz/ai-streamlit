"""Tests for PickleReader class."""

import pickle

import polars as pl
import pytest
from clustering.io.readers.pickle_reader import PickleReader


def test_pickle_reader_creation():
    """Test PickleReader initialization."""
    reader = PickleReader(path="/path/to/file.pkl")
    assert reader.path == "/path/to/file.pkl"


def test_pickle_reader_read(pickle_file, sample_data):
    """Test PickleReader read method with an actual Pickle file."""
    reader = PickleReader(path=str(pickle_file))
    result = reader.read()

    # Check that the DataFrame was read correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape

    # Check that column names are preserved
    assert result.columns == sample_data.columns

    # Check that values are correct
    for col in result.columns:
        assert result[col].to_list() == sample_data[col].to_list()


def test_pickle_reader_with_limit(temp_dir, sample_data):
    """Test PickleReader with row limiting."""
    # Create a pickle with a DataFrame
    file_path = temp_dir / "test_limit.pkl"

    # Write the sample data to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump(sample_data, f)

    # The limit functionality is applied after reading in PickleReader
    reader = PickleReader(path=str(file_path), limit=2)
    result = reader.read()

    # Check that only the specified number of rows were returned
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


def test_pickle_reader_nonexistent_file():
    """Test PickleReader with a nonexistent file."""
    reader = PickleReader(path="/path/to/nonexistent.pkl")

    # Reading a nonexistent file should raise an exception
    with pytest.raises(Exception):
        reader.read()


def test_pickle_reader_with_non_dataframe(temp_dir):
    """Test PickleReader with a non-DataFrame pickle."""
    # Create a pickle with a non-DataFrame object
    file_path = temp_dir / "test_non_df.pkl"

    # Write a list to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump([1, 2, 3, 4, 5], f)

    reader = PickleReader(path=str(file_path))

    # Should convert the list to a DataFrame
    result = reader.read()
    assert isinstance(result, pl.DataFrame)
