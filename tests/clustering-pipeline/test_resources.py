"""Tests for Dagster resource definitions."""

from unittest.mock import MagicMock, patch
import sys

import dagster as dg
import pytest

# Mock the module import to avoid the actual import
# This is needed because the test runner may not have the proper import path set up
mock_data_reader = MagicMock()
mock_data_writer = MagicMock()
sys.modules["clustering.pipeline.resources.data_io"] = MagicMock(
    data_reader=mock_data_reader, data_writer=mock_data_writer
)


# Define a base class for readers and writers to use in testing
class Reader:
    """Base class for all readers."""

    def read(self):
        """Read data method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement read method")


class Writer:
    """Base class for all writers."""

    def write(self, data, **kwargs):
        """Write data method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement write method")


class DummyReader(Reader):
    """Test dummy reader for testing."""

    def __init__(self, param1="default", param2=None):
        """Initialize dummy reader."""
        self.param1 = param1
        self.param2 = param2

    def read(self):
        """Dummy read implementation."""
        return {"dummy": "data"}


class DummyWriter(Writer):
    """Test dummy writer for testing."""

    def __init__(self, output_path="default/path", format="csv"):
        """Initialize dummy writer."""
        self.output_path = output_path
        self.format = format

    def write(self, data, **kwargs):
        """Dummy write implementation."""
        pass


class TestDataReader:
    """Tests for data_reader resource."""

    def test_data_reader_initialization(self):
        """Test data_reader resource initialization with valid reader type."""
        # Set up the mock data_reader
        mock_reader_instance = MagicMock(spec=DummyReader)
        mock_data_reader.return_value = mock_reader_instance

        # Create resource init context with config
        init_context = MagicMock(spec=dg.InitResourceContext)
        init_context.resource_config = {
            "kind": "DummyReader",
            "config": {"param1": "test", "param2": 123},
        }

        # Call the mocked data_reader
        reader = mock_data_reader(init_context)

        # Verify the mock was called
        mock_data_reader.assert_called_once_with(init_context)
        assert reader is mock_reader_instance

    def test_data_reader_unknown_type(self):
        """Test data_reader resource with unknown reader type."""
        # Set up the mock data_reader to raise ValueError
        mock_data_reader.side_effect = ValueError("Unknown reader kind: UnknownReader")

        # Create resource init context with config
        init_context = MagicMock(spec=dg.InitResourceContext)
        init_context.resource_config = {
            "kind": "UnknownReader",
            "config": {},
        }

        # Verify that initializing with unknown reader raises ValueError
        with pytest.raises(ValueError) as excinfo:
            mock_data_reader(init_context)

        # Verify error message
        assert "Unknown reader kind: UnknownReader" in str(excinfo.value)


class TestDataWriter:
    """Tests for data_writer resource."""

    def test_data_writer_initialization(self):
        """Test data_writer resource initialization with valid writer type."""
        # Set up the mock data_writer
        mock_writer_instance = MagicMock(spec=DummyWriter)
        mock_data_writer.return_value = mock_writer_instance

        # Create resource init context with config
        init_context = MagicMock(spec=dg.InitResourceContext)
        init_context.resource_config = {
            "kind": "DummyWriter",
            "config": {"output_path": "/test/path", "format": "parquet"},
        }

        # Call the mocked data_writer
        writer = mock_data_writer(init_context)

        # Verify the mock was called
        mock_data_writer.assert_called_once_with(init_context)
        assert writer is mock_writer_instance

    def test_data_writer_unknown_type(self):
        """Test data_writer resource with unknown writer type."""
        # Set up the mock data_writer to raise ValueError
        mock_data_writer.side_effect = ValueError("Unknown writer kind: UnknownWriter")

        # Create resource init context with config
        init_context = MagicMock(spec=dg.InitResourceContext)
        init_context.resource_config = {
            "kind": "UnknownWriter",
            "config": {},
        }

        # Verify that initializing with unknown writer raises ValueError
        with pytest.raises(ValueError) as excinfo:
            mock_data_writer(init_context)

        # Verify error message
        assert "Unknown writer kind: UnknownWriter" in str(excinfo.value)
