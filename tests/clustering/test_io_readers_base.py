"""Tests for clustering.io.readers.base module."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from clustering.io.readers.base import FileReader, Reader


class ConcreteReader(Reader):
    """Concrete implementation of Reader for testing."""

    def _read_from_source(self) -> pl.DataFrame:
        """Implement abstract method with test data."""
        return pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})


class ConcreteFileReader(FileReader):
    """Concrete implementation of FileReader for testing."""

    def _read_from_source(self) -> pl.DataFrame:
        """Implement abstract method with test data."""
        return pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})


class TestReader:
    """Tests for the Reader base class."""

    def test_read_returns_dataframe(self) -> None:
        """Test that read method returns a Polars DataFrame."""
        reader = ConcreteReader()
        result = reader.read()

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 2)
        assert list(result.columns) == ["col1", "col2"]

    def test_limit_parameter(self) -> None:
        """Test that limit parameter restricts the number of rows."""
        # Test with limit=2
        reader = ConcreteReader(limit=2)
        result = reader.read()

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_post_process_hook(self) -> None:
        """Test that _post_process hook is called and can modify data."""

        class PostProcessingReader(ConcreteReader):
            def _post_process(self, data: pl.DataFrame) -> pl.DataFrame:
                # Add a new column as post-processing
                return data.with_columns(pl.lit("post_processed").alias("new_col"))

        reader = PostProcessingReader()
        result = reader.read()

        assert "new_col" in result.columns
        assert result.shape == (5, 3)
        assert all(val == "post_processed" for val in result["new_col"])


class TestFileReader:
    """Tests for the FileReader base class."""

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent files."""
        # Non-existent file path
        reader = ConcreteFileReader(path="/non/existent/file.csv")

        with pytest.raises(FileNotFoundError):
            reader.read()

    def test_existing_file(self) -> None:
        """Test that an existing file passes validation."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            reader = ConcreteFileReader(path=temp_path)
            # This should not raise an exception
            result = reader.read()
            assert isinstance(result, pl.DataFrame)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_str_representation(self) -> None:
        """Test the string representation of FileReader."""
        reader = ConcreteFileReader(path="/path/to/file.csv")
        str_repr = str(reader)

        assert "ConcreteFileReader" in str_repr
        assert "/path/to/file.csv" in str_repr
