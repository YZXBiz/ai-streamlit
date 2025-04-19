"""Tests for base Writer class."""

import os
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import pytest

from clustering.io.writers.base import FileWriter, Writer


class ConcreteWriter(Writer):
    """Concrete implementation of Writer for testing."""

    data_written: ClassVar[pl.DataFrame | None] = None

    def write(self, data: pl.DataFrame) -> None:
        """Store data in a class variable for testing."""
        self.data_written = data


class ConcreteFileWriter(FileWriter):
    """Concrete implementation of FileWriter for testing."""

    data_written: ClassVar[pl.DataFrame | None] = None

    def write(self, data: pl.DataFrame) -> None:
        """Store data in a class variable for testing."""
        self._prepare_path()
        self.data_written = data


def test_writer_abstract():
    """Test that Writer is an abstract class that cannot be instantiated."""
    with pytest.raises(TypeError):
        Writer()


def test_writer_concrete(sample_data):
    """Test that a concrete implementation of Writer can be instantiated."""
    writer = ConcreteWriter()
    assert isinstance(writer, Writer)

    # Test write method
    writer.write(sample_data)
    assert writer.data_written is sample_data


def test_file_writer_str():
    """Test FileWriter string representation."""
    path = "/path/to/file.csv"
    writer = ConcreteFileWriter(path=path)

    assert writer.path == path
    assert str(writer) == f"ConcreteFileWriter(path={path})"


def test_file_writer_prepare_path(temp_dir):
    """Test that FileWriter creates parent directories."""
    nested_path = temp_dir / "subdir1" / "subdir2" / "file.csv"
    writer = ConcreteFileWriter(path=str(nested_path))

    # Call prepare_path through the write method
    writer.write(pl.DataFrame({"test": [1, 2, 3]}))

    # Check that parent directories were created
    assert Path(nested_path).parent.exists()

    # Cleanup
    os.rmdir(Path(nested_path).parent)
    os.rmdir(Path(nested_path).parent.parent)


def test_file_writer_no_create_parent_dirs(temp_dir):
    """Test FileWriter with create_parent_dirs=False."""
    nested_path = temp_dir / "no_create" / "file.csv"
    writer = ConcreteFileWriter(path=str(nested_path), create_parent_dirs=False)

    # Mock ensure_directory to verify it's not called
    with patch("clustering.io.writers.base.ensure_directory") as mock_ensure:
        writer.write(pl.DataFrame({"test": [1, 2, 3]}))
        # Ensure the method was not called
        mock_ensure.assert_not_called()
