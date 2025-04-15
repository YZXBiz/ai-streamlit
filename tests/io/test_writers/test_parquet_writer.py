"""Tests for ParquetWriter class."""

import os
from pathlib import Path

import polars as pl

from clustering.io.readers.parquet_reader import ParquetReader
from clustering.io.writers.parquet_writer import ParquetWriter


def test_parquet_writer_creation():
    """Test ParquetWriter initialization."""
    # Test with default parameters
    writer = ParquetWriter(path="/path/to/file.parquet")
    assert writer.path == "/path/to/file.parquet"
    assert writer.compression == "snappy"
    assert writer.use_pyarrow is True

    # Test with custom parameters
    writer = ParquetWriter(
        path="/path/to/file.parquet",
        compression="zstd",
        use_pyarrow=False,
    )
    assert writer.path == "/path/to/file.parquet"
    assert writer.compression == "zstd"
    assert writer.use_pyarrow is False


def test_parquet_writer_write(temp_dir, sample_data):
    """Test ParquetWriter write method."""
    file_path = temp_dir / "test_output.parquet"
    writer = ParquetWriter(path=str(file_path))

    # Write data to the file
    writer.write(sample_data)

    # Check that the file was created
    assert Path(file_path).exists()

    # Read the file back and check the contents
    reader = ParquetReader(path=str(file_path))
    result = reader.read()

    # Check that the data was written correctly
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape
    assert result.columns == sample_data.columns

    # Check that the values match (Parquet preserves types better than CSV)
    for col in result.columns:
        assert result[col].to_list() == sample_data[col].to_list()

    # Cleanup
    os.remove(file_path)


def test_parquet_writer_compression(temp_dir, sample_data):
    """Test ParquetWriter with different compression options."""
    # Test with different compression algorithms
    compressions = ["snappy", "gzip", "lz4", "zstd"]

    for compression in compressions:
        file_path = temp_dir / f"test_{compression}.parquet"
        writer = ParquetWriter(path=str(file_path), compression=compression)

        # Write data to the file
        writer.write(sample_data)

        # Check that the file was created
        assert Path(file_path).exists()

        # Read the file back and check the contents
        reader = ParquetReader(path=str(file_path))
        result = reader.read()

        # Check that the data was written correctly
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_data.shape

        # Cleanup
        os.remove(file_path)


def test_parquet_writer_create_dirs(temp_dir, sample_data):
    """Test that ParquetWriter creates parent directories."""
    nested_path = temp_dir / "parquet_subdir" / "nested.parquet"
    writer = ParquetWriter(path=str(nested_path))

    # Write data to a file in a subdirectory
    writer.write(sample_data)

    # Check that both the directory and file were created
    assert Path(nested_path).parent.exists()
    assert Path(nested_path).exists()

    # Cleanup
    os.remove(nested_path)
    os.rmdir(Path(nested_path).parent)
