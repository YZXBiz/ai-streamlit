"""IO resources for Dagster pipelines."""

from typing import Any, Dict

import dagster as dg
from pydantic import BaseModel

from clustering.infra.logging import LoggerService
from clustering.infra.monitoring import AlertingService
from clustering.io import *  # noqa


class LoggingSchema(BaseModel):
    """Schema for logging resource."""

    sink: str = "logs/dagster_log.log"
    level: str = "INFO"


@dg.resource(config_schema=LoggingSchema.model_json_schema())
def logger_service(context: dg.InitResourceContext) -> LoggerService:
    """Resource for logging.

    Args:
        context: The context for initializing the resource.

    Returns:
        LoggerService: A configured logger service.
    """
    sink = context.resource_config["sink"]
    level = context.resource_config["level"]

    logger = LoggerService(sink=sink, level=level)
    logger.start()

    @context.resource_cleanup_fn
    def cleanup_logger():
        logger.stop()

    return logger


class AlertsSchema(BaseModel):
    """Schema for alerts resource."""

    enabled: bool = True
    threshold: str = "WARNING"
    slack_webhook: str | None = None
    channels: list[str] = ["slack"]
    email_recipients: list[str] = []


@dg.resource(config_schema=AlertsSchema.model_json_schema())
def alerts_service(context: dg.InitResourceContext) -> AlertingService:
    """Resource for alerts.

    Args:
        context: The context for initializing the resource.

    Returns:
        AlertingService: A configured alerts service.
    """
    config = context.resource_config

    alerts = AlertingService(
        enabled=config.get("enabled", True),
        threshold=config.get("threshold", "WARNING"),
        channels=config.get("channels", ["slack"]),
        slack_webhook=config.get("slack_webhook"),
        email_recipients=config.get("email_recipients", []),
        alertmanager_url="http://localhost:9093",  # Default value
        logger=context.log,
    )
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
def data_reader(context: dg.InitResourceContext) -> Reader:
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
        "ParquetReader": ParquetReader,
        "CSVReader": CSVReader,
        "ExcelReader": ExcelReader,
        "PickleReader": PickleReader,
        "SnowflakeReader": SnowflakeReader,
        "BlobReader": BlobReader,
    }

    # Check if requested reader exists in our map
    reader_cls = reader_map.get(kind)

    if not reader_cls:
        raise ValueError(f"Unknown reader kind: {kind}")

    return reader_cls(**config)


class WriterSchema(BaseModel):
    """Schema for writer resource."""

    kind: str
    config: Dict[str, Any]


@dg.resource(config_schema=WriterSchema.model_json_schema())
def data_writer(context: dg.InitResourceContext) -> Writer:
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
        "ParquetWriter": ParquetWriter,
        "CSVWriter": CSVWriter,
        "ExcelWriter": ExcelWriter,
        "PickleWriter": PickleWriter,
        "SnowflakeWriter": SnowflakeWriter,
        "BlobWriter": BlobWriter,
    }

    # Check if requested writer exists in our map
    writer_cls = writer_map.get(kind)

    if not writer_cls:
        raise ValueError(f"Unknown writer kind: {kind}")

    return writer_cls(**config)
