"""Tests for ParquetReader class."""

import polars as pl
import pytest
from clustering.io.readers.parquet_reader import ParquetReader


def test_parquet_reader_creation():
    """Test ParquetReader initialization."""
    reader = ParquetReader(path="/path/to/file.parquet")
    assert reader.path == "/path/to/file.parquet"


def test_parquet_reader_read(parquet_file, sample_data):
    """Test ParquetReader read method with an actual Parquet file."""
    reader = ParquetReader(path=str(parquet_file))
    result = reader.read()

    # Check that the DataFrame was read correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape

    # Check that column names are preserved
    assert result.columns == sample_data.columns

    # Check that values are correct
    for col in result.columns:
        assert result[col].to_list() == sample_data[col].to_list()


def test_parquet_reader_with_limit(parquet_file):
    """Test ParquetReader with a row limit."""
    reader = ParquetReader(path=str(parquet_file), limit=3)
    result = reader.read()

    # Check that only the specified number of rows were read
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 3


def test_parquet_reader_nonexistent_file():
    """Test ParquetReader with a nonexistent file."""
    reader = ParquetReader(path="/path/to/nonexistent.parquet")

    # Reading a nonexistent file should raise an exception
    with pytest.raises(Exception):
        reader.read()
