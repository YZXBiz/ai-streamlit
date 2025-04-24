"""
Logging utilities for the flat_chatbot application.

This module provides a consistent interface for logging throughout the application
using Loguru for advanced logging features.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Find the project root directory (where pyproject.toml is located)
def find_project_root():
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    return Path.cwd()  # Fallback to current working directory

# Get project root
PROJECT_ROOT = find_project_root()
LOG_FILE = PROJECT_ROOT / "app.log"

# Configure default log format
DEFAULT_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Remove default logger
logger.remove()

# Add console handler with INFO level
logger.add(
    sys.stderr,
    format=DEFAULT_LOG_FORMAT,
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    colorize=True,
)

# Add file handler with DEBUG level to root directory
logger.add(
    LOG_FILE,
    format=DEFAULT_LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="1 week",  # Keep logs for 1 week
    compression="zip",  # Compress rotated logs
    backtrace=True,     # Include backtrace for errors
    diagnose=True,      # Add variables and values for errors
)


def get_logger(name: str):
    """
    Get a logger instance configured with standard settings.

    Parameters
    ----------
    name : str
        Name for the logger, typically __name__ of the calling module

    Returns
    -------
    Logger
        Configured Loguru logger instance
    """
    # With Loguru, we return the main logger with context
    # The name will be included in each log message for the module
    return logger.bind(name=name)
