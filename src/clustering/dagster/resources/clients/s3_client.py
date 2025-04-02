"""Amazon S3 client resource for Dagster."""

import os
from typing import Dict, Optional

import boto3
import dagster as dg
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel, Field


class S3ClientSchema(BaseModel):
    """Schema for S3 client configuration."""

    region_name: str = Field("us-east-1", description="AWS region name")
    endpoint_url: Optional[str] = Field(None, description="S3 endpoint URL (for LocalStack/Minio)")
    use_ssl: bool = Field(True, description="Whether to use SSL")
    verify: bool = Field(True, description="Whether to verify SSL certificates")
    aws_access_key_id: Optional[str] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS secret access key")


@dg.resource(config_schema=S3ClientSchema.model_json_schema())
def s3_client(context: dg.InitResourceContext) -> "S3ClientWrapper":
    """Resource for Amazon S3 client.

    Args:
        context: The Dagster resource initialization context

    Returns:
        S3ClientWrapper: A wrapper around boto3 S3 client with retry logic
    """
    # Get configuration values with environment variable fallback
    config = context.resource_config
    region_name = config.get("region_name", os.environ.get("AWS_REGION", "us-east-1"))
    endpoint_url = config.get("endpoint_url", os.environ.get("S3_ENDPOINT_URL"))
    use_ssl = config.get("use_ssl", True)
    verify = config.get("verify", True)
    aws_access_key_id = config.get("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = config.get("aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY"))

    # Configure client parameters
    client_kwargs = {
        "region_name": region_name,
        "use_ssl": use_ssl,
        "verify": verify,
    }

    # Add optional parameters
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key

    try:
        s3 = boto3.client("s3", **client_kwargs)
        context.log.info(f"Created S3 client for region {region_name}")
        return S3ClientWrapper(s3, context.log)
    except NoCredentialsError:
        context.log.error("No AWS credentials found")
        raise
    except Exception as e:
        context.log.error(f"Error creating S3 client: {e}")
        raise


class S3ClientWrapper:
    """Wrapper around boto3 S3 client with retry logic."""

    def __init__(self, client, logger):
        self.client = client
        self.logger = logger
        self.max_retries = 3

    def _with_retry(self, operation_name, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                last_exception = e
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                self.logger.warning(
                    f"S3 {operation_name} attempt {attempt+1}/{self.max_retries} "
                    f"failed with error code {error_code}"
                )
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"S3 {operation_name} failed after {self.max_retries} attempts")

        if last_exception:
            raise last_exception

    def upload_file(self, local_path: str, bucket: str, key: str) -> bool:
        """Upload a file to S3 with retry logic."""
        return self._with_retry("upload_file", self.client.upload_file, local_path, bucket, key)

    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """Download a file from S3 with retry logic."""
        return self._with_retry("download_file", self.client.download_file, bucket, key, local_path)

    def list_objects(self, bucket: str, prefix: str = "") -> Dict:
        """List objects in an S3 bucket with retry logic."""
        return self._with_retry("list_objects_v2", self.client.list_objects_v2, Bucket=bucket, Prefix=prefix)

    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
