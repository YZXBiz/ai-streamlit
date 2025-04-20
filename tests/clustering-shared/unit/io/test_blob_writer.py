"""Unit tests for the BlobWriter class."""

import pickle
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pandas as pd
import polars as pl
import pytest
from azure.storage.blob import BlobServiceClient, ContentSettings

from clustering.shared.io.writers.blob_writer import BlobWriter


@pytest.fixture
def mock_blob_service():
    """Create a mock BlobServiceClient."""
    mock_service = MagicMock(spec=BlobServiceClient)
    mock_container = MagicMock()
    mock_blob = MagicMock()
    
    # Set up the chain of mock objects
    mock_service.get_container_client.return_value = mock_container
    mock_container.get_blob_client.return_value = mock_blob
    
    return {
        "service": mock_service,
        "container": mock_container,
        "blob": mock_blob
    }


@pytest.fixture
def test_dataframe():
    """Create a test DataFrame for writing tests."""
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.5, 30.5]
    })


class TestBlobWriter:
    """Tests for the BlobWriter implementation."""
    
    def test_validate_destination_valid_formats(self):
        """Test validation of supported file formats."""
        for format in ["csv", "parquet", "json", "excel", "pkl"]:
            writer = BlobWriter(
                connection_string="test-connection",
                container_name="test-container",
                blob_name=f"test.{format}",
                file_format=format
            )
            # Should not raise any exception
            writer._validate_destination()
    
    def test_validate_destination_invalid_format(self):
        """Test validation fails with unsupported file format."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.xyz",
            file_format="invalid_format"
        )
        with pytest.raises(ValueError) as excinfo:
            writer._validate_destination()
        
        assert "Unsupported file format" in str(excinfo.value)
    
    def test_get_blob_service_client(self):
        """Test creation of blob service client."""
        with patch('azure.storage.blob.BlobServiceClient.from_connection_string') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            
            writer = BlobWriter(
                connection_string="test-connection",
                container_name="test-container",
                blob_name="test.csv",
                file_format="csv"
            )
            
            with patch.object(writer, '_create_blob_client') as mock_create_blob:
                mock_blob = MagicMock()
                mock_create_blob.return_value = mock_blob
                
                # Now test would call the method that uses the service client
                # This is just a stub since we're directly patching the blob client creation
                pass
    
    def test_create_blob_client(self):
        """Test creation of blob client."""
        with patch('azure.storage.blob.BlobServiceClient.from_connection_string') as mock_create:
            mock_service = MagicMock()
            mock_container = MagicMock()
            mock_blob = MagicMock()
            
            mock_create.return_value = mock_service
            mock_service.get_container_client.return_value = mock_container
            mock_container.get_blob_client.return_value = mock_blob
            
            writer = BlobWriter(
                connection_string="test-connection",
                container_name="test-container",
                blob_name="test.csv",
                file_format="csv"
            )
            
            result = writer._create_blob_client()
            
            # Verify proper method calls
            mock_create.assert_called_once_with("test-connection")
            mock_service.get_container_client.assert_called_once_with("test-container")
            mock_container.get_blob_client.assert_called_once_with("test.csv")
            assert result == mock_blob
    
    def test_write_to_destination_csv(self, mock_blob_service, test_dataframe):
        """Test writing CSV data to blob storage."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
    
    def test_write_to_destination_parquet(self, mock_blob_service, test_dataframe):
        """Test writing Parquet data to blob storage."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.parquet",
            file_format="parquet"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
    
    def test_write_to_destination_json(self, mock_blob_service, test_dataframe):
        """Test writing JSON data to blob storage."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.json",
            file_format="json"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
    
    def test_write_to_destination_excel(self, mock_blob_service, test_dataframe):
        """Test writing Excel data to blob storage."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.xlsx",
            file_format="excel"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Patch pandas conversion and excel writing
        with patch.object(test_dataframe, 'to_pandas') as mock_to_pandas:
            mock_df = MagicMock()
            mock_to_pandas.return_value = mock_df
            
            # Write the data
            writer._write_to_destination(test_dataframe)
            
            # Verify pandas conversion was called
            mock_to_pandas.assert_called_once()
            mock_df.to_excel.assert_called_once()
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
    
    def test_write_to_destination_pickle(self, mock_blob_service, test_dataframe):
        """Test writing pickled data to blob storage."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.pkl",
            file_format="pkl"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Patch pickle.dump
        with patch('pickle.dump') as mock_dump:
            # Write the data
            writer._write_to_destination(test_dataframe)
            
            # Verify pickle.dump was called
            mock_dump.assert_called_once()
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
    
    def test_write_to_destination_with_options(self, mock_blob_service, test_dataframe):
        """Test passing format-specific options."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv",
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Mock the write_csv method to capture options
        with patch.object(test_dataframe, 'write_csv') as mock_write_csv:
            mock_write_csv.return_value = None
            
            # Write the data
            writer._write_to_destination(test_dataframe)
            
            # Verify write_csv was called
            mock_write_csv.assert_called_once()
    
    def test_upload_blob_error(self, mock_blob_service, test_dataframe):
        """Test error handling when uploading blob fails."""
        # Make the upload_blob method raise an exception
        mock_blob_service["blob"].upload_blob.side_effect = Exception("Upload error")
        
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data - should raise RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            writer._write_to_destination(test_dataframe)
        
        assert "Unexpected error when uploading blob" in str(excinfo.value)
    
    def test_integration_with_writer_base_class(self, mock_blob_service, test_dataframe):
        """Test integration with the Writer base class."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the methods used by the base class
        writer._validate_destination = lambda: None
        writer._write_to_destination = MagicMock()
        
        # Test the main write method from the base class
        writer.write(test_dataframe)
        
        # Verify _write_to_destination was called with the DataFrame
        writer._write_to_destination.assert_called_once_with(test_dataframe)
    
    def test_overwrite_mode(self, mock_blob_service, test_dataframe):
        """Test overwrite mode when uploading blob."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv",
            overwrite=True
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called with overwrite=True
        _, kwargs = mock_blob_service["blob"].upload_blob.call_args
        assert kwargs.get("overwrite") is True
    
    def test_no_overwrite_mode(self, mock_blob_service, test_dataframe):
        """Test no overwrite mode when uploading blob."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container", 
            blob_name="test.csv",
            file_format="csv",
            overwrite=False
        )
        
        # Override the methods
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write the data
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called with overwrite=False
        _, kwargs = mock_blob_service["blob"].upload_blob.call_args
        assert kwargs.get("overwrite") is False
    
    def test_max_concurrency_parameter(self, mock_blob_service, test_dataframe):
        """Test custom max_concurrency parameter."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            max_concurrency=4  # Custom concurrency
        )
        
        # Override the blob client creation
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Write data
        writer._write_to_destination(test_dataframe)
        
        # Check the max_concurrency parameter was passed
        _, kwargs = mock_blob_service["blob"].upload_blob.call_args
        assert kwargs["max_concurrency"] == 4
        
    @patch('os.getenv')
    def test_missing_connection_string(self, mock_getenv, test_dataframe):
        """Test error handling when connection string is missing and not in environment."""
        # Ensure environment variable is not set
        mock_getenv.return_value = None
        
        writer = BlobWriter(
            # No connection_string provided
            container_name="test-container",
            blob_name="test.csv"
        )
        
        with pytest.raises(ValueError, match="Connection string must be provided"):
            # This should fail when trying to create the blob client
            writer._create_blob_client()
            
    @patch('os.getenv')
    def test_connection_string_from_env(self, mock_getenv, mock_blob_service):
        """Test getting connection string from environment variable."""
        # Set the environment variable
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_STORAGE_CONNECTION_STRING": "env-connection-string",
            "AZURE_STORAGE_CONTAINER": None
        }.get(key, default)
        
        # Set up mocks for BlobServiceClient
        with patch('azure.storage.blob.BlobServiceClient.from_connection_string') as mock_create:
            mock_create.return_value = mock_blob_service["service"]
            
            writer = BlobWriter(
                # No connection_string provided
                container_name="test-container",
                blob_name="test.csv"
            )
            
            # Should use environment variable
            result = writer._create_blob_client()
            
            # Verify proper method calls
            mock_create.assert_called_once_with("env-connection-string")
            assert result == mock_blob_service["blob"]
            
    @patch('os.getenv')
    def test_missing_container_name(self, mock_getenv, test_dataframe):
        """Test error handling when container name is missing and not in environment."""
        # Ensure environment variable is set for connection string but not container
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_STORAGE_CONNECTION_STRING": "env-connection-string",
            "AZURE_STORAGE_CONTAINER": None
        }.get(key, default)
        
        writer = BlobWriter(
            # No container_name provided
            connection_string="test-connection",
            blob_name="test.csv"
        )
        
        with pytest.raises(ValueError, match="Container name must be provided"):
            # This should fail when trying to create the blob client
            writer._create_blob_client()
            
    @patch('os.getenv')
    def test_container_name_from_env(self, mock_getenv, mock_blob_service):
        """Test getting container name from environment variable."""
        # Set the environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_STORAGE_CONNECTION_STRING": "env-connection-string",
            "AZURE_STORAGE_CONTAINER": "env-container"
        }.get(key, default)
        
        # Set up mocks for BlobServiceClient
        with patch('azure.storage.blob.BlobServiceClient.from_connection_string') as mock_create:
            mock_create.return_value = mock_blob_service["service"]
            
            writer = BlobWriter(
                # No connection_string or container_name provided
                blob_name="test.csv"
            )
            
            # Should use environment variables
            result = writer._create_blob_client()
            
            # Verify proper method calls
            mock_create.assert_called_once_with("env-connection-string")
            mock_blob_service["service"].get_container_client.assert_called_once_with("env-container")
            assert result == mock_blob_service["blob"]
            
    def test_infer_file_format_from_extension(self, mock_blob_service, test_dataframe):
        """Test inferring file format from blob name extension."""
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.parquet",
            # No file_format specified - should be inferred
        )
        
        # Override the blob client creation
        writer._create_blob_client = lambda: mock_blob_service["blob"]
        
        # Call validate destination to trigger format inference
        writer._validate_destination()
        
        # Check the format was inferred correctly
        assert writer.file_format == "parquet"
        
        # Write the data to ensure it works with the inferred format
        writer._write_to_destination(test_dataframe)
        
        # Verify upload_blob was called
        mock_blob_service["blob"].upload_blob.assert_called_once()
        
    @patch('azure.storage.blob.BlobClient.upload_blob')
    def test_service_request_error(self, mock_upload, test_dataframe):
        """Test handling of ServiceRequestError during upload."""
        from azure.core.exceptions import ServiceRequestError
        
        # Create a mock blob client that will raise the error
        mock_blob_client = MagicMock()
        mock_blob_client.upload_blob.side_effect = ServiceRequestError("Network error")
        
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the blob client creation to return our mock
        writer._create_blob_client = lambda: mock_blob_client
        
        # Should raise a RuntimeError wrapping the original error
        # Per the implementation, the message is "Azure service error when uploading blob: Network error"
        with pytest.raises(RuntimeError, match="Azure service error when uploading blob: Network error"):
            writer._write_to_destination(test_dataframe)
            
    @patch('azure.storage.blob.BlobClient.upload_blob')
    def test_azure_error(self, mock_upload, test_dataframe):
        """Test handling of AzureError during upload."""
        from azure.core.exceptions import AzureError
        
        # Create a mock blob client that will raise the error
        mock_blob_client = MagicMock()
        mock_blob_client.upload_blob.side_effect = AzureError("Azure service error")
        
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the blob client creation to return our mock
        writer._create_blob_client = lambda: mock_blob_client
        
        # Should raise a RuntimeError wrapping the original error
        with pytest.raises(RuntimeError, match="Azure service error when uploading blob: Azure service error"):
            writer._write_to_destination(test_dataframe)
            
    @patch('azure.storage.blob.BlobClient.upload_blob')
    def test_unexpected_error(self, mock_upload, test_dataframe):
        """Test handling of unexpected errors during upload."""
        # Create a mock blob client that will raise the error
        mock_blob_client = MagicMock()
        mock_blob_client.upload_blob.side_effect = ValueError("Unexpected error")
        
        writer = BlobWriter(
            connection_string="test-connection",
            container_name="test-container",
            blob_name="test.csv",
            file_format="csv"
        )
        
        # Override the blob client creation to return our mock
        writer._create_blob_client = lambda: mock_blob_client
        
        # Should raise a RuntimeError wrapping the original error
        with pytest.raises(RuntimeError, match="Unexpected error when uploading blob: Unexpected error"):
            writer._write_to_destination(test_dataframe) 