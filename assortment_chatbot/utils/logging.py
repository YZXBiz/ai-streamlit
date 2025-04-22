"""
Logging configuration for the Assortment Chatbot application.

This module sets up Loguru as the primary logging system and provides
helper functions for consistent logging across the application.
"""

import logging as python_logging
import os
import sys
from pathlib import Path

from loguru import logger
from loguru._logger import Logger

# Default log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure default sink to stderr
logger.remove()  # Remove default sink
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)

# Add file logging
logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="1 week",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Add error file logging for errors only
logger.add(
    LOG_DIR / "error.log",
    rotation="10 MB",
    retention="1 month",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
)


class InterceptHandler(python_logging.Handler):
    """
    Intercepts standard library logging and redirects to Loguru.

    This handler intercepts all standard logging calls and redirects them
    to Loguru, ensuring all logs go through the same system.
    """

    def emit(self, record: python_logging.LogRecord) -> None:
        """
        Intercept log records and pass them to Loguru.

        Args:
            record: The standard logging record to intercept
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where this was logged
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == python_logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str | None = None) -> None:
    """
    Configure application-wide logging settings.

    Args:
        level: Optional logging level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Set log level from environment or parameter
    log_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.configure(levels=[{"name": "TRACE", "no": 5, "color": "<cyan>"}])

    # Set level for all sinks
    for handler_id in logger._core.handlers:
        logger._core.handlers[handler_id]._levelno = logger.level(log_level).no

    # Intercept standard library logging
    python_logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Intercept third-party logs
    for logger_name in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "streamlit",
        "snowflake",
        "duckdb",
        "asyncio",
    ]:
        lib_logger = python_logging.getLogger(logger_name)
        lib_logger.handlers = [InterceptHandler()]
        lib_logger.propagate = False


def get_logger(name: str) -> Logger:
    """
    Get a configured logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        A configured Loguru logger instance
    """
    return logger.bind(name=name)


# Initialize as soon as this module is imported
setup_logging()
