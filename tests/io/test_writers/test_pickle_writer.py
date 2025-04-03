"""Tests for PickleWriter class."""

import os
import pickle
from pathlib import Path

import polars as pl
from clustering.io.readers.pickle_reader import PickleReader
from clustering.io.writers.pickle_writer import PickleWriter


def test_pickle_writer_creation():
    """Test PickleWriter initialization."""
    # Test with default parameters
    writer = PickleWriter(path="/path/to/file.pkl")
    assert writer.path == "/path/to/file.pkl"
    assert writer.protocol == pickle.HIGHEST_PROTOCOL

    # Test with custom parameters
    writer = PickleWriter(
        path="/path/to/file.pkl",
        protocol=pickle.DEFAULT_PROTOCOL,
    )
    assert writer.path == "/path/to/file.pkl"
    assert writer.protocol == pickle.DEFAULT_PROTOCOL


def test_pickle_writer_write(temp_dir, sample_data):
    """Test PickleWriter write method."""
    file_path = temp_dir / "test_output.pkl"
    writer = PickleWriter(path=str(file_path))

    # Write data to the file
    writer.write(sample_data)

    # Check that the file was created
    assert Path(file_path).exists()

    # Read the file back and check the contents
    reader = PickleReader(path=str(file_path))
    result = reader.read()

    # Check that the data was written correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape
    assert result.columns == sample_data.columns

    # Check that the values match
    for col in result.columns:
        assert result[col].to_list() == sample_data[col].to_list()

    # Cleanup
    os.remove(file_path)


def test_pickle_writer_protocols(temp_dir, sample_data):
    """Test PickleWriter with different pickle protocols."""
    # Test with different protocols
    protocols = [2, 3, 4, pickle.HIGHEST_PROTOCOL]

    for protocol in protocols:
        file_path = temp_dir / f"test_protocol_{protocol}.pkl"
        writer = PickleWriter(path=str(file_path), protocol=protocol)

        # Write data to the file
        writer.write(sample_data)

        # Check that the file was created
        assert Path(file_path).exists()

        # Directly load with pickle to verify
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Check that the data was written correctly
        assert isinstance(loaded_data, pl.DataFrame)
        assert loaded_data.shape == sample_data.shape

        # Cleanup
        os.remove(file_path)


def test_pickle_writer_create_dirs(temp_dir, sample_data):
    """Test that PickleWriter creates parent directories."""
    nested_path = temp_dir / "pickle_subdir" / "nested.pkl"
    writer = PickleWriter(path=str(nested_path))

    # Write data to a file in a subdirectory
    writer.write(sample_data)

    # Check that both the directory and file were created
    assert Path(nested_path).parent.exists()
    assert Path(nested_path).exists()

    # Cleanup
    os.remove(nested_path)
    os.rmdir(Path(nested_path).parent)
