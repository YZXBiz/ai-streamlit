"""Azure Blob Storage reader implementation."""

import os
import pickle
from io import BytesIO

import polars as pl
from azure.storage.blob import BlobClient

from clustering.io.readers.base import Reader


class BlobReader(Reader):
    """Reader for Azure Blob Storage.

    Supports reading CSV, Parquet, and Pickle files from Azure Blob Storage.
    """

    blob_name: str
    max_concurrency: int = 8

    def read(self) -> pl.DataFrame:
        """Read data from Azure Blob Storage.

        Returns:
            DataFrame containing the data
        """
        # Create blob client
        account_url = os.getenv("ACCOUNT_URL", "https://account.blob.core.windows.net")
        container_name = os.getenv("CONTAINER_NAME", "container")

        blob_client = BlobClient(
            account_url=account_url,
            container_name=container_name,
            blob_name=self.blob_name,
            credential=None,  # Use DefaultAzureCredential by default
        )

        try:
            # Download the blob content with parallelism
            blob_data = blob_client.download_blob(max_concurrency=self.max_concurrency).readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob: {e}")

        # Determine the file format from the blob name extension
        file_extension = os.path.splitext(self.blob_name)[1].lower()

        # Process based on file type
        if file_extension == ".csv":
            data = pl.read_csv(BytesIO(blob_data))
        elif file_extension == ".parquet":
            data = pl.read_parquet(BytesIO(blob_data))
        elif file_extension in [".pkl", ".pickle"]:
            data = pickle.loads(blob_data)
            # Convert to Polars DataFrame if needed
            if not isinstance(data, pl.DataFrame):
                data = pl.from_pandas(data) if hasattr(data, "to_pandas") else pl.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Please use a CSV, Parquet, or Pickle file.")

        if self.limit is not None:
            data = data.head(self.limit)

        return data
