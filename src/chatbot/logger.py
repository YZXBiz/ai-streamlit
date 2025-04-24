"""
Logging utilities for the assortment_chatbot application.

This module provides a consistent interface for logging throughout the application.
"""

import logging
import os
import sys
from logging import Logger

# Configure logging format
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(name: str) -> Logger:
    """
    Get a logger instance configured with standard settings.

    Parameters
    ----------
    name : str
        Name for the logger, typically __name__ of the calling module

    Returns
    -------
    Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set up console handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Set log level based on environment or default to INFO
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
