"""Azure Blob Storage reader implementation."""

import pickle
from io import BytesIO

import polars as pl
from azure.storage.blob import BlobServiceClient

from clustering.shared.io.readers.base import Reader


class BlobReader(Reader):
    """Reader for Azure Blob Storage.

    Supports reading CSV, Parquet, and Pickle files from Azure Blob Storage.
    """

    connection_string: str
    container_name: str
    blob_path: str
    file_format: str
    max_concurrency: int = 8

    def _validate_source(self) -> None:
        """Validate the blob storage connection and file format.

        Raises:
            ValueError: If the file format is not supported
        """
        valid_formats = ["csv", "parquet", "json", "excel", "pickle"]
        if self.file_format not in valid_formats:
            raise ValueError(
                f"Unsupported file format: {self.file_format}. "
                f"Supported formats are: {', '.join(valid_formats)}"
            )

    def _get_blob_service_client(self) -> BlobServiceClient:
        """Get the Azure Blob Service client.

        Returns:
            BlobServiceClient: The Azure Blob Service client
        """
        return BlobServiceClient.from_connection_string(self.connection_string)

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from Azure Blob Storage.

        Returns:
            DataFrame containing the data

        Raises:
            RuntimeError: If there's an error downloading from blob storage
        """
        # Get blob service client
        blob_service_client = self._get_blob_service_client()

        # Get container client
        container_client = blob_service_client.get_container_client(self.container_name)

        # Get blob client
        blob_client = container_client.get_blob_client(self.blob_path)

        try:
            # Download the blob content with parallelism
            blob_data = blob_client.download_blob(max_concurrency=self.max_concurrency).readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob: {e}")

        # Process based on file format specified
        if self.file_format == "csv":
            data = pl.read_csv(BytesIO(blob_data))
        elif self.file_format == "parquet":
            data = pl.read_parquet(BytesIO(blob_data))
        elif self.file_format == "json":
            data = pl.read_json(BytesIO(blob_data))
        elif self.file_format == "excel":
            data = pl.read_excel(BytesIO(blob_data))
        elif self.file_format == "pickle":
            data = pickle.loads(blob_data)
            # Convert to Polars DataFrame if needed
            if not isinstance(data, pl.DataFrame):
                data = pl.from_pandas(data) if hasattr(data, "to_pandas") else pl.DataFrame(data)
        else:
            # This should never happen due to validation in _validate_source
            raise ValueError(f"Unsupported file format: {self.file_format}")

        return data
