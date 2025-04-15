"""Azure Blob Storage writer implementation."""

import os
import pickle
from io import BytesIO

import polars as pl
from azure.core.exceptions import AzureError, ServiceRequestError
from azure.storage.blob import BlobClient

from clustering.io.writers.base import Writer


class BlobWriter(Writer):
    """Writer for Azure Blob Storage.

    Support writing CSV, Parquet, and Pickle files to Azure Blob Storage.
    """

    blob_name: str
    overwrite: bool = True
    max_concurrency: int = 8

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to Azure Blob Storage.

        Args:
            data: DataFrame to write

        Raises:
            ValueError: If the file format is not supported
            RuntimeError: If there's an error uploading to blob storage
        """
        buffer = BytesIO()
        file_extension = os.path.splitext(self.blob_name)[1].lower()

        # Write to buffer based on file extension
        if file_extension == ".csv":
            data.write_csv(buffer)
        elif file_extension == ".parquet":
            data.write_parquet(buffer)
        elif file_extension in [".pkl", ".pickle"]:
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("Unsupported file format. Please use .csv, .parquet, .pkl, or .pickle")

        # Reset buffer position
        buffer.seek(0)

        # Create blob client
        account_url = os.getenv("ACCOUNT_URL", "https://account.blob.core.windows.net")
        container_name = os.getenv("CONTAINER_NAME", "container")

        blob_client = BlobClient(
            account_url=account_url,
            container_name=container_name,
            blob_name=self.blob_name,
            credential=None,  # Use DefaultAzureCredential by default
        )

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
