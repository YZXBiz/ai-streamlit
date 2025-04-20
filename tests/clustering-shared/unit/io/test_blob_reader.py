"""Unit tests for the BlobReader class."""

import pickle
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from azure.storage.blob import BlobServiceClient

from clustering.shared.io.readers.blob_reader import BlobReader


@pytest.fixture
def mock_blob_service():
    """Create a mock BlobServiceClient."""
    mock_service = MagicMock(spec=BlobServiceClient)
    mock_container = MagicMock()
    mock_blob = MagicMock()
    
    # Set up the chain of mock objects
    mock_service.get_container_client.return_value = mock_container
    mock_container.get_blob_client.return_value = mock_blob
    
    # Create a mock download object
    mock_download = MagicMock()
    
    # Default response is CSV data
    mock_download.readall.return_value = b"id,name,value\n1,Alice,10.5\n2,Bob,20.5\n3,Charlie,30.5"
    mock_blob.download_blob.return_value = mock_download
    
    return {
        "service": mock_service,
        "container": mock_container,
        "blob": mock_blob,
        "download": mock_download
    }


class TestBlobReader:
    """Tests for the BlobReader implementation."""
    
    def test_validate_source_valid_formats(self):
        """Test validation of supported file formats."""
        for format in ["csv", "parquet", "json", "excel", "pickle"]:
            reader = BlobReader(
                connection_string="test-connection",
                container_name="test-container",
                blob_path="test.csv",
                file_format=format
            )
            # Should not raise any exception
            reader._validate_source()
    
    def test_validate_source_invalid_format(self):
        """Test validation fails with unsupported file format."""
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="invalid_format"
        )
        with pytest.raises(ValueError) as excinfo:
            reader._validate_source()
        
        assert "Unsupported file format" in str(excinfo.value)
        assert "csv, parquet, json, excel, pickle" in str(excinfo.value)
    
    def test_get_blob_service_client(self):
        """Test creation of blob service client."""
        with patch('azure.storage.blob.BlobServiceClient.from_connection_string') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            reader = BlobReader(
                connection_string="test-connection",
                container_name="test-container",
                blob_path="test.csv",
                file_format="csv"
            )
            
            result = reader._get_blob_service_client()
            
            # Verify client was created with connection string
            mock_create.assert_called_once_with("test-connection")
            assert result == mock_client
    
    def test_read_from_source_csv(self, mock_blob_service):
        """Test reading CSV data from blob storage."""
        # Set up the mock to return CSV data
        mock_blob_service["download"].readall.return_value = b"id,name,value\n1,Alice,10.5\n2,Bob,20.5"
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        result = reader._read_from_source()
        
        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "value"]
        
        # Verify proper method calls
        mock_blob_service["service"].get_container_client.assert_called_once_with("test-container")
        mock_blob_service["container"].get_blob_client.assert_called_once_with("test.csv")
        mock_blob_service["blob"].download_blob.assert_called_once()
    
    def test_read_from_source_parquet(self, mock_blob_service):
        """Test reading Parquet data from blob storage."""
        # Create a simple parquet file in memory
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        parquet_bytes = BytesIO()
        df.to_parquet(parquet_bytes)
        parquet_bytes.seek(0)
        
        # Set the mock to return parquet data
        mock_blob_service["download"].readall.return_value = parquet_bytes.getvalue()
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.parquet",
            file_format="parquet"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        with patch('polars.read_parquet', return_value=pl.from_pandas(df)) as mock_read:
            result = reader._read_from_source()
            
            # Verify polars.read_parquet was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)
    
    def test_read_from_source_json(self, mock_blob_service):
        """Test reading JSON data from blob storage."""
        # Set the mock to return JSON data
        mock_blob_service["download"].readall.return_value = b'[{"id":1,"name":"Alice","value":10.5},{"id":2,"name":"Bob","value":20.5}]'
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.json",
            file_format="json"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        with patch('polars.read_json', return_value=pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})) as mock_read:
            result = reader._read_from_source()
            
            # Verify polars.read_json was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)
    
    def test_read_from_source_excel(self, mock_blob_service):
        """Test reading Excel data from blob storage."""
        # Set the mock to return binary data (we'll mock the actual reading)
        mock_blob_service["download"].readall.return_value = b"excel_data"
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.xlsx",
            file_format="excel"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        with patch('polars.read_excel', return_value=pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})) as mock_read:
            result = reader._read_from_source()
            
            # Verify polars.read_excel was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)
    
    def test_read_from_source_pickle_polars_df(self, mock_blob_service):
        """Test reading pickled Polars DataFrame from blob storage."""
        # Create a pickled Polars DataFrame
        df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        pickled_data = pickle.dumps(df)
        
        mock_blob_service["download"].readall.return_value = pickled_data
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        result = reader._read_from_source()
        
        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "value"]
    
    def test_read_from_source_pickle_pandas_df(self, mock_blob_service):
        """Test reading pickled Pandas DataFrame from blob storage."""
        # Create a pickled Pandas DataFrame
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        pickled_data = pickle.dumps(df)
        
        mock_blob_service["download"].readall.return_value = pickled_data
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        result = reader._read_from_source()
        
        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns
    
    def test_read_from_source_pickle_dict(self, mock_blob_service):
        """Test reading pickled dictionary from blob storage."""
        # Create a pickled dictionary
        data_dict = {"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]}
        pickled_data = pickle.dumps(data_dict)
        
        mock_blob_service["download"].readall.return_value = pickled_data
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        result = reader._read_from_source()
        
        # Verify the result
        assert isinstance(result, pl.DataFrame)
    
    def test_blob_download_error(self, mock_blob_service):
        """Test error handling when downloading blob fails."""
        # Make the download blob method raise an exception
        mock_blob_service["blob"].download_blob.side_effect = Exception("Connection error")
        
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv"
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data - should raise RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            reader._read_from_source()
        
        assert "Failed to download blob" in str(excinfo.value)
    
    def test_max_concurrency_parameter(self, mock_blob_service):
        """Test that max_concurrency parameter is passed correctly."""
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
            max_concurrency=16
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read the data
        reader._read_from_source()
        
        # Verify the max_concurrency parameter was passed
        mock_blob_service["blob"].download_blob.assert_called_once_with(max_concurrency=16)
    
    def test_integration_with_reader_base_class(self, mock_blob_service):
        """Test integration with the Reader base class."""
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
            limit=1  # Test limit functionality from base class
        )
        
        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]
        
        # Read with base class template method
        result = reader.read()
        
        # Should have applied limit from base class
        assert len(result) == 1 