"""Tests for base Reader class."""

import polars as pl
import pytest
from clustering.io.readers.base import FileReader, Reader


class ConcreteReader(Reader):
    """Concrete implementation of Reader for testing."""

    def read(self) -> pl.DataFrame:
        """Return a mock DataFrame."""
        return pl.DataFrame({"test": [1, 2, 3]})


class ConcreteFileReader(FileReader):
    """Concrete implementation of FileReader for testing."""

    def read(self) -> pl.DataFrame:
        """Return a mock DataFrame."""
        return pl.DataFrame({"test": [1, 2, 3]})


def test_reader_abstract():
    """Test that Reader is an abstract class that cannot be instantiated."""
    with pytest.raises(TypeError):
        Reader()


def test_reader_concrete():
    """Test that a concrete implementation of Reader can be instantiated."""
    reader = ConcreteReader()
    assert isinstance(reader, Reader)

    # Test read method
    result = reader.read()
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 1)


def test_file_reader():
    """Test FileReader implementation."""
    path = "/path/to/file.csv"
    reader = ConcreteFileReader(path=path)

    assert reader.path == path
    assert str(reader) == f"ConcreteFileReader(path={path})"

    # Test read method
    result = reader.read()
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 1)
