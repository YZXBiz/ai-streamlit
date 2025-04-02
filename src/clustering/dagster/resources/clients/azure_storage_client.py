"""Azure Blob Storage client resource for Dagster."""

import os
import time
from typing import BinaryIO, Dict, List, Optional, Union

import dagster as dg
from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError  # type: ignore
from azure.storage.blob import BlobServiceClient  # type: ignore
from pydantic import BaseModel, Field, SecretStr


class AzureStorageClientSchema(BaseModel):
    """Schema for Azure Blob Storage client configuration."""

    connection_string: SecretStr = Field(None, description="Azure Storage connection string")
    account_name: str = Field(None, description="Azure Storage account name")
    account_key: SecretStr = Field(None, description="Azure Storage account key")
    sas_token: SecretStr = Field(None, description="Azure Storage SAS token")
    endpoint_suffix: str = Field("core.windows.net", description="Azure Storage endpoint suffix")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay: float = Field(1.0, description="Base delay between retries in seconds")


@dg.resource(config_schema=AzureStorageClientSchema.model_json_schema())
def azure_storage_client(context: dg.InitResourceContext) -> "AzureStorageClientWrapper":
    """Resource for Azure Blob Storage client.

    Args:
        context: The Dagster resource initialization context

    Returns:
        AzureStorageClientWrapper: A wrapper around Azure Blob Storage client with retry logic
    """
    # Get configuration values with environment variable fallback
    config = context.resource_config

    # Connection can be established using either connection string or account + credentials
    connection_string = config.get("connection_string")
    if connection_string and isinstance(connection_string, SecretStr):
        connection_string = connection_string.get_secret_value()
    else:
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

    account_name = config.get("account_name", os.environ.get("AZURE_STORAGE_ACCOUNT"))

    account_key = config.get("account_key")
    if account_key and isinstance(account_key, SecretStr):
        account_key = account_key.get_secret_value()
    else:
        account_key = os.environ.get("AZURE_STORAGE_KEY")

    sas_token = config.get("sas_token")
    if sas_token and isinstance(sas_token, SecretStr):
        sas_token = sas_token.get_secret_value()
    else:
        sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")

    endpoint_suffix = config.get("endpoint_suffix", "core.windows.net")
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 1.0)

    logger = context.log

    try:
        # Create the client based on available credentials
        if connection_string:
            if logger:
                logger.info("Creating Azure Blob Storage client using connection string")
            client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            if logger:
                logger.info(f"Creating Azure Blob Storage client for account {account_name}")
            client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.{endpoint_suffix}", credential=account_key
            )
        elif account_name and sas_token:
            if logger:
                logger.info(f"Creating Azure Blob Storage client for account {account_name} with SAS token")
            client = BlobServiceClient(account_url=f"https://{account_name}.blob.{endpoint_suffix}?{sas_token}")
        else:
            error_msg = "Either connection string or account name with credentials must be provided"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

        return AzureStorageClientWrapper(client, max_retries, retry_delay, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error initializing Azure Blob Storage client: {e}")
        raise


class AzureStorageClientWrapper:
    """Wrapper around Azure Blob Storage client with retry logic."""

    def __init__(self, client: BlobServiceClient, max_retries: int, retry_delay: float, logger):
        """Initialize the Azure Blob Storage client wrapper.

        Args:
            client: Azure Blob Storage service client
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            logger: Logger object
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger

    def _with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute an operation with retry logic.

        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Result of the operation

        Raises:
            Exception: If the operation fails after max_retries attempts
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = operation_func(*args, **kwargs)
                return result

            except (ResourceNotFoundError, ResourceExistsError) as e:
                # Don't retry if resource doesn't exist or already exists
                if self.logger:
                    self.logger.error(f"Non-retryable Azure error: {str(e)}")
                raise

            except AzureError as e:
                last_exception = e
                if self.logger:
                    self.logger.warning(
                        f"Azure {operation_name} attempt {attempt+1}/{self.max_retries} " f"failed with error: {str(e)}"
                    )

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2**attempt) + (0.1 * attempt)
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"Azure {operation_name} failed after {self.max_retries} attempts")

        if last_exception:
            raise last_exception

    def list_containers(self) -> List[str]:
        """List all containers in the storage account.

        Returns:
            List of container names
        """

        def _list_containers():
            return [container["name"] for container in self.client.list_containers()]

        return self._with_retry("list_containers", _list_containers)

    def container_exists(self, container_name: str) -> bool:
        """Check if a container exists.

        Args:
            container_name: Name of the container

        Returns:
            True if the container exists, False otherwise
        """

        def _container_exists(container_name):
            try:
                self.client.get_container_client(container_name).get_container_properties()
                return True
            except ResourceNotFoundError:
                return False

        return self._with_retry("container_exists", _container_exists, container_name)

    def create_container(self, container_name: str, public_access: Optional[str] = None) -> None:
        """Create a new container.

        Args:
            container_name: Name of the container
            public_access: Public access level (None, 'container', or 'blob')
        """

        def _create_container(container_name, public_access):
            self.client.create_container(container_name, public_access=public_access)

        self._with_retry("create_container", _create_container, container_name, public_access)

    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[Dict]:
        """List blobs in a container.

        Args:
            container_name: Name of the container
            prefix: Filter results to blobs with this prefix

        Returns:
            List of blob properties dictionaries
        """

        def _list_blobs(container_name, prefix):
            container_client = self.client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [{"name": blob.name, "size": blob.size, "last_modified": blob.last_modified} for blob in blobs]

        return self._with_retry("list_blobs", _list_blobs, container_name, prefix)

    def upload_blob(
        self, container_name: str, blob_name: str, data: Union[str, bytes, BinaryIO], overwrite: bool = False
    ) -> Dict:
        """Upload data to a blob.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            data: Content to upload (string, bytes, or file-like object)
            overwrite: Whether to overwrite existing blob

        Returns:
            Dictionary of blob properties
        """

        def _upload_blob(container_name, blob_name, data, overwrite):
            blob_client = self.client.get_blob_client(container_name, blob_name)
            result = blob_client.upload_blob(data, overwrite=overwrite)
            return {
                "container": container_name,
                "blob": blob_name,
                "etag": result.get("etag"),
                "last_modified": result.get("last_modified"),
            }

        return self._with_retry("upload_blob", _upload_blob, container_name, blob_name, data, overwrite)

    def download_blob(self, container_name: str, blob_name: str) -> bytes:
        """Download a blob's content.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob

        Returns:
            Blob content as bytes
        """

        def _download_blob(container_name, blob_name):
            blob_client = self.client.get_blob_client(container_name, blob_name)
            download = blob_client.download_blob()
            return download.readall()

        return self._with_retry("download_blob", _download_blob, container_name, blob_name)

    def download_blob_to_file(self, container_name: str, blob_name: str, file_path: str) -> None:
        """Download a blob to a local file.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            file_path: Local file path to save the blob content
        """

        def _download_to_file(container_name, blob_name, file_path):
            blob_client = self.client.get_blob_client(container_name, blob_name)
            with open(file_path, "wb") as file:
                download = blob_client.download_blob()
                file.write(download.readall())

        self._with_retry("download_blob_to_file", _download_to_file, container_name, blob_name, file_path)

    def upload_file(self, container_name: str, blob_name: str, file_path: str, overwrite: bool = False) -> Dict:
        """Upload a local file to a blob.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            file_path: Local file path to upload
            overwrite: Whether to overwrite existing blob

        Returns:
            Dictionary of blob properties
        """

        def _upload_file(container_name, blob_name, file_path, overwrite):
            blob_client = self.client.get_blob_client(container_name, blob_name)
            with open(file_path, "rb") as file:
                result = blob_client.upload_blob(file, overwrite=overwrite)
                return {
                    "container": container_name,
                    "blob": blob_name,
                    "etag": result.get("etag"),
                    "last_modified": result.get("last_modified"),
                }

        return self._with_retry("upload_file", _upload_file, container_name, blob_name, file_path, overwrite)

    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob

        Returns:
            True if the blob exists, False otherwise
        """

        def _blob_exists(container_name, blob_name):
            try:
                blob_client = self.client.get_blob_client(container_name, blob_name)
                blob_client.get_blob_properties()
                return True
            except ResourceNotFoundError:
                return False

        return self._with_retry("blob_exists", _blob_exists, container_name, blob_name)

    def delete_blob(self, container_name: str, blob_name: str) -> None:
        """Delete a blob.

        Args:
            container_name: Name of the container
            blob_name: Name of the blob
        """

        def _delete_blob(container_name, blob_name):
            blob_client = self.client.get_blob_client(container_name, blob_name)
            blob_client.delete_blob()

        self._with_retry("delete_blob", _delete_blob, container_name, blob_name)
