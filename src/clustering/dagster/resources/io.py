"""Resources for I/O operations in Dagster pipelines."""

from typing import Any, Dict

import dagster as dg
from pydantic import BaseModel

from clustering.io import datasets, services


class LoggingSchema(BaseModel):
    """Schema for logging resource."""

    sink: str = "logs/dagster_log.log"
    level: str = "INFO"


@dg.resource(config_schema=LoggingSchema.model_json_schema())
def logger_service(context: dg.InitResourceContext) -> services.LoggerService:
    """Resource for logging.

    Args:
        context: The context for initializing the resource.

    Returns:
        LoggerService: A configured logger service.
    """
    sink = context.resource_config["sink"]
    level = context.resource_config["level"]

    logger = services.LoggerService(sink=sink, level=level)
    logger.start()

    @context.resource_cleanup_fn
    def cleanup_logger():
        logger.stop()

    return logger


class AlertsSchema(BaseModel):
    """Schema for alerts resource."""

    enabled: bool = True
    threshold: str = "WARNING"


@dg.resource(config_schema=AlertsSchema.model_json_schema())
def alerts_service(context: dg.InitResourceContext) -> services.AlertsService:
    """Resource for alerts.

    Args:
        context: The context for initializing the resource.

    Returns:
        AlertsService: A configured alerts service.
    """
    enabled = context.resource_config["enabled"]
    threshold = context.resource_config["threshold"]

    alerts = services.AlertsService(enabled=enabled, threshold=threshold)
    alerts.start()

    @context.resource_cleanup_fn
    def cleanup_alerts():
        alerts.stop()

    return alerts


class ReaderSchema(BaseModel):
    """Schema for reader resource."""

    kind: str
    config: Dict[str, Any]


@dg.resource(config_schema=ReaderSchema.model_json_schema())
def data_reader(context: dg.InitResourceContext) -> datasets.Reader:
    """Resource for reading data.

    Args:
        context: The context for initializing the resource.

    Returns:
        Reader: A configured reader.
    """
    kind = context.resource_config["kind"]
    config = context.resource_config["config"]

    # Create reader based on kind
    reader_map = {
        "ParquetReader": datasets.ParquetReader,
        "ExcelReader": datasets.ExcelReader,
        "CSVReader": datasets.CSVReader,
        "PickleReader": datasets.PickleReader,
        "SnowflakeReader": datasets.SnowflakeReader,
        "BlobReader": datasets.BlobReader,
    }

    reader_cls = reader_map.get(kind)
    if not reader_cls:
        raise ValueError(f"Unknown reader kind: {kind}")

    return reader_cls(**config)


class WriterSchema(BaseModel):
    """Schema for writer resource."""

    kind: str
    config: Dict[str, Any]


@dg.resource(config_schema=WriterSchema.model_json_schema())
def data_writer(context: dg.InitResourceContext) -> datasets.Writer:
    """Resource for writing data.

    Args:
        context: The context for initializing the resource.

    Returns:
        Writer: A configured writer.
    """
    kind = context.resource_config["kind"]
    config = context.resource_config["config"]

    # Create writer based on kind
    writer_map = {
        "ParquetWriter": datasets.ParquetWriter,
        "ExcelWriter": datasets.ExcelWriter,
        "CSVWriter": datasets.CSVWriter,
        "PickleWriter": datasets.PickleWriter,
        "SnowflakeWriter": datasets.SnowflakeWriter,
        "BlobWriter": datasets.BlobWriter,
    }

    writer_cls = writer_map.get(kind)
    if not writer_cls:
        raise ValueError(f"Unknown writer kind: {kind}")

    return writer_cls(**config)
