"""Tests for ExcelWriter class."""

import os
from pathlib import Path

import polars as pl

from clustering.io.readers.excel_reader import ExcelReader
from clustering.io.writers.excel_writer import ExcelWriter


def test_excel_writer_creation():
    """Test ExcelWriter initialization."""
    # Test with default parameters
    writer = ExcelWriter(path="/path/to/file.xlsx")
    assert writer.path == "/path/to/file.xlsx"
    assert writer.sheet_name == "Sheet1"
    assert writer.engine == "openpyxl"

    # Test with custom parameters
    writer = ExcelWriter(
        path="/path/to/file.xlsx",
        sheet_name="CustomSheet",
        engine="openpyxl",
    )
    assert writer.path == "/path/to/file.xlsx"
    assert writer.sheet_name == "CustomSheet"
    assert writer.engine == "openpyxl"


def test_excel_writer_write(temp_dir, sample_data):
    """Test ExcelWriter write method."""
    file_path = temp_dir / "test_output.xlsx"
    writer = ExcelWriter(path=str(file_path))

    # Write data to the file
    writer.write(sample_data)

    # Check that the file was created
    assert Path(file_path).exists()

    # Read the file back and check the contents
    reader = ExcelReader(path=str(file_path))
    result = reader.read()

    # Check that the data was written correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == sample_data.shape[0]

    # Excel reading/writing can sometimes introduce additional columns or change data types,
    # so we check for basic structure rather than exact equality
    assert all(col in result.columns for col in sample_data.columns)

    # Cleanup
    os.remove(file_path)


def test_excel_writer_custom_sheet(temp_dir, sample_data):
    """Test ExcelWriter with a custom sheet name."""
    file_path = temp_dir / "test_custom_sheet.xlsx"
    sheet_name = "TestSheet"
    writer = ExcelWriter(path=str(file_path), sheet_name=sheet_name)

    # Write data to the file
    writer.write(sample_data)

    # Check that the file was created
    assert Path(file_path).exists()

    # Read the file back with the correct sheet name
    reader = ExcelReader(path=str(file_path), sheet_name=sheet_name)
    result = reader.read()

    # Check that the data was written correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == sample_data.shape[0]

    # Cleanup
    os.remove(file_path)


def test_excel_writer_create_dirs(temp_dir, sample_data):
    """Test that ExcelWriter creates parent directories."""
    nested_path = temp_dir / "excel_subdir" / "nested.xlsx"
    writer = ExcelWriter(path=str(nested_path))

    # Write data to a file in a subdirectory
    writer.write(sample_data)

    # Check that both the directory and file were created
    assert Path(nested_path).parent.exists()
    assert Path(nested_path).exists()

    # Cleanup
    os.remove(nested_path)
    os.rmdir(Path(nested_path).parent)
