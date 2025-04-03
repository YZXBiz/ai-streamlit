"""Tests for ExcelReader class."""

import polars as pl
import pytest
from clustering.io.readers.excel_reader import ExcelReader


def test_excel_reader_creation():
    """Test ExcelReader initialization."""
    # Test with default parameters
    reader = ExcelReader(path="/path/to/file.xlsx")
    assert reader.path == "/path/to/file.xlsx"
    assert reader.sheet_name == 0
    assert reader.engine == "openpyxl"

    # Test with custom parameters
    reader = ExcelReader(
        path="/path/to/file.xlsx",
        sheet_name="Sheet1",
        engine="openpyxl",
    )
    assert reader.path == "/path/to/file.xlsx"
    assert reader.sheet_name == "Sheet1"
    assert reader.engine == "openpyxl"


def test_excel_reader_read(excel_file, sample_data):
    """Test ExcelReader read method with an actual Excel file."""
    reader = ExcelReader(path=str(excel_file))
    result = reader.read()

    # Check that the DataFrame was read correctly
    assert isinstance(result, pl.DataFrame)

    # Excel reading can sometimes introduce additional columns or change data types,
    # so we check for basic structure rather than exact equality
    assert result.shape[0] == sample_data.shape[0]
    assert all(col in result.columns for col in sample_data.columns)


def test_excel_reader_with_limit(excel_file):
    """Test ExcelReader with a row limit."""
    reader = ExcelReader(path=str(excel_file), limit=2)
    result = reader.read()

    # Check that only the first 2 rows were read
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 2


def test_excel_reader_nonexistent_file():
    """Test ExcelReader with a nonexistent file."""
    reader = ExcelReader(path="/path/to/nonexistent.xlsx")

    # Reading a nonexistent file should raise an exception
    with pytest.raises(Exception):
        reader.read()
