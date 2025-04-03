"""Specialized data source resources for the clustering pipeline."""

from typing import Any, Dict, Optional

import dagster as dg
from pydantic import BaseModel, Field

# Import client resources
from clustering.io import BlobReader, BlobWriter, CSVReader, ParquetReader, SnowflakeReader, SnowflakeWriter


class SalesDataReaderSchema(BaseModel):
    """Configuration for sales data reader."""

    source_type: str = Field("parquet", description="Source type (parquet, csv, snowflake, azure_blob)")
    path: Optional[str] = Field(None, description="File path for file-based sources")
    query: Optional[str] = Field(None, description="Query for database sources")
    container: Optional[str] = Field(None, description="Container name for blob storage")
    blob_path: Optional[str] = Field(None, description="Blob path for blob storage")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options for the reader")


@dg.resource(config_schema=SalesDataReaderSchema.model_json_schema())
def sales_data_reader(context: dg.InitResourceContext):
    """Resource for reading sales data from various sources.

    Args:
        context: The resource initialization context

    Returns:
        A configured reader object with a read() method
    """
    config = context.resource_config
    source_type = config.get("source_type", "parquet")
    options = config.get("options", {})

    context.log.info(f"Initializing sales data reader of type: {source_type}")

    if source_type == "csv":
        path = config.get("path")
        if not path:
            raise ValueError("Path is required for csv source type")
        reader = CSVReader(path=path, **options)

    elif source_type == "parquet":
        path = config.get("path")
        if not path:
            raise ValueError("Path is required for parquet source type")
        reader = ParquetReader(path=path, **options)

    elif source_type == "snowflake":
        # Use snowflake client if available
        if hasattr(context.resources, "snowflake_client"):
            snowflake = context.resources.snowflake_client
            query = config.get("query")
            if not query:
                raise ValueError("Query is required for snowflake source type")

            reader = SnowflakeReader(client=snowflake, query=query, **options)
        else:
            raise ValueError("Snowflake client resource is required but not available")

    elif source_type == "azure_blob":
        # Use azure storage client if available
        if hasattr(context.resources, "azure_storage"):
            azure = context.resources.azure_storage
            container = config.get("container")
            blob_path = config.get("blob_path")

            if not container or not blob_path:
                raise ValueError("Container and blob_path are required for azure_blob source type")

            reader = BlobReader(client=azure, container=container, blob_path=blob_path, **options)
        else:
            raise ValueError("Azure storage resource is required but not available")
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    return reader


class NeedStateDataReaderSchema(BaseModel):
    """Configuration for need state data reader."""

    source_type: str = Field("csv", description="Source type (csv, parquet, snowflake, azure_blob)")
    path: Optional[str] = Field(None, description="File path for file-based sources")
    query: Optional[str] = Field(None, description="Query for database sources")
    container: Optional[str] = Field(None, description="Container name for blob storage")
    blob_path: Optional[str] = Field(None, description="Blob path for blob storage")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options for the reader")


@dg.resource(config_schema=NeedStateDataReaderSchema.model_json_schema())
def need_state_data_reader(context: dg.InitResourceContext):
    """Resource for reading need state data from various sources.

    Args:
        context: The resource initialization context

    Returns:
        A configured reader object with a read() method
    """
    config = context.resource_config
    source_type = config.get("source_type", "csv")
    options = config.get("options", {})

    context.log.info(f"Initializing need state data reader of type: {source_type}")

    if source_type == "csv":
        path = config.get("path")
        if not path:
            raise ValueError("Path is required for csv source type")
        reader = CSVReader(path=path, **options)

    elif source_type == "parquet":
        path = config.get("path")
        if not path:
            raise ValueError("Path is required for parquet source type")
        reader = ParquetReader(path=path, **options)

    elif source_type == "snowflake":
        # Use snowflake client if available
        if hasattr(context.resources, "snowflake_client"):
            snowflake = context.resources.snowflake_client
            query = config.get("query")
            if not query:
                raise ValueError("Query is required for snowflake source type")

            reader = SnowflakeReader(client=snowflake, query=query, **options)
        else:
            raise ValueError("Snowflake client resource is required but not available")

    elif source_type == "azure_blob":
        # Use azure storage client if available
        if hasattr(context.resources, "azure_storage"):
            azure = context.resources.azure_storage
            container = config.get("container")
            blob_path = config.get("blob_path")

            if not container or not blob_path:
                raise ValueError("Container and blob_path are required for azure_blob source type")

            reader = BlobReader(client=azure, container=container, blob_path=blob_path, **options)
        else:
            raise ValueError("Azure storage resource is required but not available")
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    return reader


class DataWriterSchema(BaseModel):
    """Configuration for data writer."""

    destination_type: str = Field("parquet", description="Destination type (parquet, csv, snowflake, azure_blob)")
    path: Optional[str] = Field(None, description="File path for file-based destinations")
    table_name: Optional[str] = Field(None, description="Table name for database destinations")
    container: Optional[str] = Field(None, description="Container name for blob storage")
    blob_path: Optional[str] = Field(None, description="Blob path for blob storage")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options for the writer")


@dg.resource(config_schema=DataWriterSchema.model_json_schema())
def data_writer(context: dg.InitResourceContext):
    """Resource for writing data to various destinations.

    Args:
        context: The resource initialization context

    Returns:
        A configured writer object with a write() method
    """
    config = context.resource_config
    destination_type = config.get("destination_type", "parquet")
    options = config.get("options", {})

    context.log.info(f"Initializing data writer of type: {destination_type}")

    if destination_type == "parquet":
        from clustering.io import ParquetWriter

        path = config.get("path")
        if not path:
            raise ValueError("Path is required for parquet destination type")
        writer = ParquetWriter(path=path, **options)

    elif destination_type == "csv":
        from clustering.io import CSVWriter

        path = config.get("path")
        if not path:
            raise ValueError("Path is required for csv destination type")
        writer = CSVWriter(path=path, **options)

    elif destination_type == "snowflake":
        # Use snowflake client if available
        if hasattr(context.resources, "snowflake_client"):
            snowflake = context.resources.snowflake_client
            table_name = config.get("table_name")
            if not table_name:
                raise ValueError("Table name is required for snowflake destination type")

            writer = SnowflakeWriter(client=snowflake, table_name=table_name, **options)
        else:
            raise ValueError("Snowflake client resource is required but not available")

    elif destination_type == "azure_blob":
        # Use azure storage client if available
        if hasattr(context.resources, "azure_storage"):
            azure = context.resources.azure_storage
            container = config.get("container")
            blob_path = config.get("blob_path")

            if not container or not blob_path:
                raise ValueError("Container and blob_path are required for azure_blob destination type")

            writer = BlobWriter(client=azure, container=container, blob_path=blob_path, **options)
        else:
            raise ValueError("Azure storage resource is required but not available")
    else:
        raise ValueError(f"Unknown destination type: {destination_type}")

    return writer
