"""Azure Blob Storage writer implementation."""

import os
import pickle
from io import BytesIO
from typing import Optional

import polars as pl
from azure.core.exceptions import AzureError, ServiceRequestError
from azure.storage.blob import BlobClient, BlobServiceClient

from clustering.shared.io.writers.base import Writer


class BlobWriter(Writer):
    """Writer for Azure Blob Storage.

    Support writing CSV, Parquet, and Pickle files to Azure Blob Storage.
    """

    connection_string: Optional[str] = None
    container_name: Optional[str] = None
    blob_name: str
    file_format: Optional[str] = None
    overwrite: bool = True
    max_concurrency: int = 8

    def _validate_destination(self) -> None:
        """Validate the blob storage parameters.

        Raises:
            ValueError: If the file format is not supported
        """
        # If file_format is not specified, infer it from blob_name extension
        if self.file_format is None:
            file_extension = os.path.splitext(self.blob_name)[1].lower()
            if file_extension.startswith("."):
                self.file_format = file_extension[1:]

        # Validate file format
        valid_formats = ["csv", "parquet", "json", "excel", "pkl", "pickle"]
        if self.file_format not in valid_formats:
            raise ValueError(
                f"Unsupported file format: {self.file_format}. "
                f"Supported formats are: {', '.join(valid_formats)}"
            )

    def _create_blob_client(self) -> BlobClient:
        """Create a blob client.

        Returns:
            BlobClient: The Azure Blob Client

        Raises:
            ValueError: If required parameters are missing
        """
        # Get connection string from environment if not provided
        conn_string = self.connection_string
        if conn_string is None:
            conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_string:
                raise ValueError(
                    "Connection string must be provided or set in AZURE_STORAGE_CONNECTION_STRING"
                )

        # Get container name from environment if not provided
        container = self.container_name
        if container is None:
            container = os.getenv("AZURE_STORAGE_CONTAINER")
            if not container:
                raise ValueError(
                    "Container name must be provided or set in AZURE_STORAGE_CONTAINER"
                )

        # Create blob client using connection string approach
        blob_service = BlobServiceClient.from_connection_string(conn_string)
        container_client = blob_service.get_container_client(container)
        return container_client.get_blob_client(self.blob_name)

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to Azure Blob Storage.

        Args:
            data: DataFrame to write

        Raises:
            ValueError: If the file format is not supported
            RuntimeError: If there's an error uploading to blob storage
        """
        # Validate destination before writing
        self._validate_destination()

        # Create buffer to hold data
        buffer = BytesIO()

        # Write to buffer based on file format
        if self.file_format == "csv":
            data.write_csv(buffer)
        elif self.file_format == "parquet":
            data.write_parquet(buffer)
        elif self.file_format == "json":
            data.write_json(buffer)
        elif self.file_format in ["pkl", "pickle"]:
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.file_format == "excel":
            # Convert to pandas for Excel writing
            pandas_df = data.to_pandas()
            pandas_df.to_excel(buffer, index=False)
        else:
            # This should never happen due to validation
            raise ValueError(f"Unsupported file format: {self.file_format}")

        # Reset buffer position
        buffer.seek(0)

        # Create blob client
        blob_client = self._create_blob_client()

        # Upload to blob storage
        try:
            blob_client.upload_blob(
                buffer,
                blob_type="BlockBlob",
                overwrite=self.overwrite,
                max_concurrency=self.max_concurrency,
            )
        except AzureError as e:
            raise RuntimeError(f"Azure service error when uploading blob: {str(e)}") from e
        except ServiceRequestError as e:
            raise RuntimeError(f"Network error when uploading blob: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when uploading blob: {str(e)}") from e
