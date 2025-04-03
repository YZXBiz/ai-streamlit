"""Tests for CSVReader class."""

import polars as pl
from clustering.io.readers.csv_reader import CSVReader


def test_csv_reader_creation():
    """Test CSVReader initialization."""
    # Test with default parameters
    reader = CSVReader(path="/path/to/file.csv")
    assert reader.path == "/path/to/file.csv"
    assert reader.delimiter == ","
    assert reader.has_header is True

    # Test with custom parameters
    reader = CSVReader(
        path="/path/to/file.csv",
        delimiter="|",
        has_header=False,
    )
    assert reader.path == "/path/to/file.csv"
    assert reader.delimiter == "|"
    assert reader.has_header is False


def test_csv_reader_read(csv_file, sample_data):
    """Test CSVReader read method with an actual CSV file."""
    reader = CSVReader(path=str(csv_file))
    result = reader.read()

    # Check that the DataFrame was read correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape

    # Check that column names are preserved
    assert result.columns == sample_data.columns

    # Check that values are correct
    for col in result.columns:
        assert result[col].to_list() == sample_data[col].to_list()


def test_csv_reader_with_limit(csv_file):
    """Test CSVReader with a row limit."""
    reader = CSVReader(path=str(csv_file), limit=2)
    result = reader.read()

    # Check that only the first 2 rows were read
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


def test_csv_reader_with_delimiter(temp_dir, sample_data):
    """Test CSVReader with a custom delimiter."""
    # Create a CSV with a different delimiter
    file_path = temp_dir / "test_pipe.csv"
    sample_data.write_csv(file_path, separator="|")

    # Read with the correct delimiter
    reader = CSVReader(path=str(file_path), delimiter="|")
    result = reader.read()

    # Check that the data was read correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape

    # Read with the wrong delimiter (should still parse but data will be incorrect)
    reader_wrong = CSVReader(path=str(file_path), delimiter=",")
    result_wrong = reader_wrong.read()

    # The result should be different when using the wrong delimiter
    assert result_wrong.shape != result.shape or result_wrong.columns != result.columns
