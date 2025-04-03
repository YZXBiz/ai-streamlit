"""Tests for Azure Blob Storage reader and writer with mocked connections."""

import pickle
from io import BytesIO
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from azure.storage.blob import BlobClient
from clustering.io.readers.blob_reader import BlobReader
from clustering.io.writers.blob_writer import BlobWriter


@pytest.fixture
def mock_blob_client():
    """Create a mock Azure BlobClient."""
    mock_client = MagicMock(spec=BlobClient)
    return mock_client


@pytest.fixture
def mock_blob_download_stream(sample_data):
    """Create a mock download stream for blob data."""
    mock_stream = MagicMock()

    # Create different data formats that the BlobReader might expect
    csv_buffer = BytesIO()
    sample_data.write_csv(csv_buffer)
    csv_buffer.seek(0)

    parquet_buffer = BytesIO()
    sample_data.write_parquet(parquet_buffer)
    parquet_buffer.seek(0)

    pickle_buffer = BytesIO()
    pickle.dump(sample_data, pickle_buffer)
    pickle_buffer.seek(0)

    # Store the data for different file types
    mock_stream.file_data = {
        ".csv": csv_buffer.getvalue(),
        ".parquet": parquet_buffer.getvalue(),
        ".pkl": pickle_buffer.getvalue(),
    }

    # Configure readall to return the appropriate data
    mock_stream.readall.return_value = mock_stream.file_data[".parquet"]

    return mock_stream


def test_blob_reader_init():
    """Test BlobReader initialization."""
    reader = BlobReader(blob_name="test_data.parquet", max_concurrency=10)

    assert reader.blob_name == "test_data.parquet"
    assert reader.max_concurrency == 10


@patch("clustering.io.readers.blob_reader.BlobClient")
def test_blob_reader_read_parquet(mock_blob_client_class, mock_blob_client, mock_blob_download_stream, sample_data):
    """Test BlobReader read method with Parquet files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client
    mock_blob_client.download_blob.return_value = mock_blob_download_stream
    mock_blob_download_stream.readall.return_value = mock_blob_download_stream.file_data[".parquet"]

    # Create reader and read data
    reader = BlobReader(blob_name="test_data.parquet")
    result = reader.read()

    # Verify BlobClient was created and download_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.download_blob.assert_called_once_with(max_concurrency=8)

    # Check result
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape
    assert result.columns == sample_data.columns


@patch("clustering.io.readers.blob_reader.BlobClient")
def test_blob_reader_read_csv(mock_blob_client_class, mock_blob_client, mock_blob_download_stream, sample_data):
    """Test BlobReader read method with CSV files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client
    mock_blob_client.download_blob.return_value = mock_blob_download_stream
    mock_blob_download_stream.readall.return_value = mock_blob_download_stream.file_data[".csv"]

    # Create reader and read data
    reader = BlobReader(blob_name="test_data.csv")
    result = reader.read()

    # Verify BlobClient was created and download_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.download_blob.assert_called_once_with(max_concurrency=8)

    # Check result
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape
    assert result.columns == sample_data.columns


@patch("clustering.io.readers.blob_reader.BlobClient")
def test_blob_reader_read_pickle(mock_blob_client_class, mock_blob_client, mock_blob_download_stream, sample_data):
    """Test BlobReader read method with Pickle files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client
    mock_blob_client.download_blob.return_value = mock_blob_download_stream
    mock_blob_download_stream.readall.return_value = mock_blob_download_stream.file_data[".pkl"]

    # Create reader and read data
    reader = BlobReader(blob_name="test_data.pkl")
    result = reader.read()

    # Verify BlobClient was created and download_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.download_blob.assert_called_once_with(max_concurrency=8)

    # Check result
    assert isinstance(result, pl.DataFrame)
    assert result.shape == sample_data.shape
    assert result.columns == sample_data.columns


@patch("clustering.io.readers.blob_reader.BlobClient")
def test_blob_reader_invalid_extension(mock_blob_client_class, mock_blob_client, mock_blob_download_stream):
    """Test BlobReader with an invalid file extension."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client
    mock_blob_client.download_blob.return_value = mock_blob_download_stream

    # Create reader with invalid extension
    reader = BlobReader(blob_name="test_data.txt")

    # Reading an unsupported file type should raise an exception
    with pytest.raises(ValueError):
        reader.read()


def test_blob_writer_init():
    """Test BlobWriter initialization."""
    # Test with default parameters
    writer = BlobWriter(blob_name="test_data.parquet")
    assert writer.blob_name == "test_data.parquet"
    assert writer.overwrite is True
    assert writer.max_concurrency == 8

    # Test with custom parameters
    writer = BlobWriter(blob_name="test_data.parquet", overwrite=False, max_concurrency=10)
    assert writer.blob_name == "test_data.parquet"
    assert writer.overwrite is False
    assert writer.max_concurrency == 10


@patch("clustering.io.writers.blob_writer.BlobClient")
def test_blob_writer_write_parquet(mock_blob_client_class, mock_blob_client, sample_data):
    """Test BlobWriter write method with Parquet files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client

    # Create writer and write data
    writer = BlobWriter(blob_name="test_data.parquet")
    writer.write(sample_data)

    # Verify BlobClient was created and upload_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.upload_blob.assert_called_once()

    # Check that upload_blob was called with correct parameters
    args, kwargs = mock_blob_client.upload_blob.call_args
    assert kwargs["blob_type"] == "BlockBlob"
    assert kwargs["overwrite"] is True
    assert kwargs["max_concurrency"] == 8


@patch("clustering.io.writers.blob_writer.BlobClient")
def test_blob_writer_write_csv(mock_blob_client_class, mock_blob_client, sample_data):
    """Test BlobWriter write method with CSV files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client

    # Create writer and write data
    writer = BlobWriter(blob_name="test_data.csv")
    writer.write(sample_data)

    # Verify BlobClient was created and upload_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.upload_blob.assert_called_once()


@patch("clustering.io.writers.blob_writer.BlobClient")
def test_blob_writer_write_pickle(mock_blob_client_class, mock_blob_client, sample_data):
    """Test BlobWriter write method with Pickle files."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client

    # Create writer and write data
    writer = BlobWriter(blob_name="test_data.pkl")
    writer.write(sample_data)

    # Verify BlobClient was created and upload_blob was called
    mock_blob_client_class.assert_called_once()
    mock_blob_client.upload_blob.assert_called_once()


@patch("clustering.io.writers.blob_writer.BlobClient")
def test_blob_writer_invalid_extension(mock_blob_client_class, mock_blob_client, sample_data):
    """Test BlobWriter with an invalid file extension."""
    # Configure mocks
    mock_blob_client_class.return_value = mock_blob_client

    # Create writer with invalid extension
    writer = BlobWriter(blob_name="test_data.txt")

    # Writing an unsupported file type should raise an exception
    with pytest.raises(ValueError):
        writer.write(sample_data)
