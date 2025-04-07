"""Logging resources for Dagster pipelines."""

import dagster as dg

from clustering.infra.logging import LoggerService


@dg.resource(
    config_schema={
        "sink": dg.Field(
            dg.String, default_value="logs/dagster_log.log", description="Path to log file"
        ),
        "level": dg.Field(dg.String, default_value="INFO", description="Log level"),
    }
)
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
