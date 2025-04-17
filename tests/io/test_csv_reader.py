"""Tests for the CSV reader module."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from clustering.io.readers.csv_reader import CSVReader


@pytest.fixture
def sample_csv_file() -> Path:
    """Create a temporary CSV file for testing.
    
    Returns:
        Path to the temporary CSV file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
        
        # Write sample data
        temp.write(b"id,name,value\n")
        temp.write(b"1,Item 1,10.5\n")
        temp.write(b"2,Item 2,20.75\n")
        temp.write(b"3,Item 3,30.25\n")
    
    # Return the path to the file
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        os.unlink(temp_path)


@pytest.fixture
def sample_csv_file_with_quotes() -> Path:
    """Create a temporary CSV file with quoted fields for testing.
    
    Returns:
        Path to the temporary CSV file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
        
        # Write sample data with quotes
        temp.write(b'id,name,description\n')
        temp.write(b'1,"Item 1","This is a, quoted description"\n')
        temp.write(b'2,"Item 2","Another, quoted description"\n')
        temp.write(b'3,"Item 3","Final, quoted description"\n')
    
    # Return the path to the file
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        os.unlink(temp_path)


@pytest.fixture
def sample_tsv_file() -> Path:
    """Create a temporary TSV file for testing.
    
    Returns:
        Path to the temporary TSV file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as temp:
        temp_path = Path(temp.name)
        
        # Write sample data with tab delimiter
        temp.write(b"id\tname\tvalue\n")
        temp.write(b"1\tItem 1\t10.5\n")
        temp.write(b"2\tItem 2\t20.75\n")
        temp.write(b"3\tItem 3\t30.25\n")
    
    # Return the path to the file
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        os.unlink(temp_path)


def test_csv_reader_basic(sample_csv_file: Path) -> None:
    """Test basic CSV reader functionality.
    
    Args:
        sample_csv_file: Path to a sample CSV file.
    """
    # Create CSV reader
    reader = CSVReader(path=str(sample_csv_file))
    
    # Read data
    df = reader.read()
    
    # Assert shape
    assert df.shape == (3, 3)
    
    # Assert column names
    assert df.columns == ["id", "name", "value"]
    
    # Assert data types
    assert df.dtypes[0] == pl.Int64
    assert df.dtypes[1] == pl.Utf8
    assert df.dtypes[2] == pl.Float64
    
    # Assert values
    assert df.select("id").to_series().to_list() == [1, 2, 3]
    assert df.select("name").to_series().to_list() == ["Item 1", "Item 2", "Item 3"]
    assert df.select("value").to_series().to_list() == [10.5, 20.75, 30.25]


def test_csv_reader_with_limit(sample_csv_file: Path) -> None:
    """Test CSV reader with a row limit.
    
    Args:
        sample_csv_file: Path to a sample CSV file.
    """
    # Create CSV reader with limit
    reader = CSVReader(path=str(sample_csv_file), limit=2)
    
    # Read data
    df = reader.read()
    
    # Assert shape (only 2 rows due to limit)
    assert df.shape == (2, 3)
    
    # Assert values
    assert df.select("id").to_series().to_list() == [1, 2]
    assert df.select("name").to_series().to_list() == ["Item 1", "Item 2"]


def test_csv_reader_with_quotes(sample_csv_file_with_quotes: Path) -> None:
    """Test CSV reader with quoted fields.
    
    Args:
        sample_csv_file_with_quotes: Path to a sample CSV file with quoted fields.
    """
    # Create CSV reader
    reader = CSVReader(path=str(sample_csv_file_with_quotes))
    
    # Read data
    df = reader.read()
    
    # Assert shape
    assert df.shape == (3, 3)
    
    # Assert values with embedded commas are correctly parsed
    descriptions = df.select("description").to_series().to_list()
    assert "This is a, quoted description" in descriptions
    assert "Another, quoted description" in descriptions
    assert "Final, quoted description" in descriptions


def test_csv_reader_with_tsv(sample_tsv_file: Path) -> None:
    """Test CSV reader with TSV (tab-separated) file.
    
    Args:
        sample_tsv_file: Path to a sample TSV file.
    """
    # Create CSV reader with tab delimiter
    reader = CSVReader(path=str(sample_tsv_file), delimiter="\t")
    
    # Read data
    df = reader.read()
    
    # Assert shape
    assert df.shape == (3, 3)
    
    # Assert column names
    assert df.columns == ["id", "name", "value"]
    
    # Assert values
    assert df.select("id").to_series().to_list() == [1, 2, 3]
    assert df.select("name").to_series().to_list() == ["Item 1", "Item 2", "Item 3"]
    assert df.select("value").to_series().to_list() == [10.5, 20.75, 30.25]


def test_csv_reader_nonexistent_file() -> None:
    """Test CSV reader with a nonexistent file."""
    # Create CSV reader with nonexistent file
    reader = CSVReader(path="nonexistent_file.csv")
    
    # Reading should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        reader.read()


def test_csv_reader_empty_file() -> None:
    """Test CSV reader with an empty file."""
    # Create a temporary empty file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
        # Write minimal header to prevent NoDataError
        temp.write(b"column1,column2\n")
    
    try:
        # Create CSV reader with empty file
        reader = CSVReader(path=str(temp_path))
        
        # Reading should not raise error but return DataFrame with columns and no rows
        df = reader.read()
        assert df.shape[0] == 0
        assert len(df.columns) > 0
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path) 