"""Tests for the CSV writer module."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from clustering.io.writers.csv_writer import CSVWriter


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for testing.

    Returns:
        A sample polars DataFrame.
    """
    return pl.DataFrame(
        {"id": [1, 2, 3], "name": ["Item 1", "Item 2", "Item 3"], "value": [10.5, 20.75, 30.25]}
    )


def test_csv_writer_basic(sample_dataframe: pl.DataFrame) -> None:
    """Test basic CSV writer functionality.

    Args:
        sample_dataframe: A sample DataFrame to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create CSV writer
        writer = CSVWriter(path=str(temp_path))

        # Write data
        writer.write(sample_dataframe)

        # Check file exists
        assert os.path.exists(temp_path)

        # Read back the data to verify
        df_read = pl.read_csv(temp_path)

        # Assert shape
        assert df_read.shape == (3, 3)

        # Assert column names
        assert df_read.columns == ["id", "name", "value"]

        # Assert values
        assert df_read.select("id").to_series().to_list() == [1, 2, 3]
        assert df_read.select("name").to_series().to_list() == ["Item 1", "Item 2", "Item 3"]
        assert df_read.select("value").to_series().to_list() == [10.5, 20.75, 30.25]
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_csv_writer_with_custom_delimiter(sample_dataframe: pl.DataFrame) -> None:
    """Test CSV writer with custom delimiter.

    Args:
        sample_dataframe: A sample DataFrame to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create CSV writer with tab delimiter
        writer = CSVWriter(path=str(temp_path), delimiter="\t")

        # Write data
        writer.write(sample_dataframe)

        # Read back the data with tab delimiter to verify
        df_read = pl.read_csv(temp_path, separator="\t")

        # Assert shape
        assert df_read.shape == (3, 3)

        # Assert values
        assert df_read.select("id").to_series().to_list() == [1, 2, 3]
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_csv_writer_without_header(sample_dataframe: pl.DataFrame) -> None:
    """Test CSV writer without header.

    Args:
        sample_dataframe: A sample DataFrame to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create CSV writer without header
        writer = CSVWriter(path=str(temp_path), include_header=False)

        # Write data
        writer.write(sample_dataframe)

        # Read back the data without header to verify
        df_read = pl.read_csv(temp_path, has_header=False, new_columns=["col1", "col2", "col3"])

        # Assert shape
        assert df_read.shape == (3, 3)

        # Assert values (first row should be the first data row, not header)
        assert df_read.select("col1").to_series().to_list() == [1, 2, 3]
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_csv_writer_create_parent_dirs() -> None:
    """Test CSV writer creates parent directories."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define a nested path that doesn't exist yet
        nested_dir = Path(temp_dir) / "subdir1" / "subdir2"
        file_path = nested_dir / "output.csv"

        # Create sample data
        df = pl.DataFrame({"a": [1, 2, 3]})

        # Create CSV writer with parent directory creation
        writer = CSVWriter(path=str(file_path), create_parent_dirs=True)

        # Write data
        writer.write(df)

        # Check file and directories exist
        assert nested_dir.exists()
        assert file_path.exists()


def test_csv_writer_empty_dataframe() -> None:
    """Test CSV writer with empty DataFrame."""
    # Create empty DataFrame
    empty_df = pl.DataFrame({"a": [], "b": []})

    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create CSV writer
        writer = CSVWriter(path=str(temp_path))

        # Writing empty DataFrame should raise ValueError
        with pytest.raises(ValueError, match="Cannot write empty DataFrame"):
            writer.write(empty_df)
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)
