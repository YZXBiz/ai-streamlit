"""
Tests for logger module.
"""

import logging
import os
from unittest.mock import patch

import pytest

from chatbot.logger import get_logger


@pytest.fixture
def reset_logging() -> None:
    """Reset logging configuration before each test."""
    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Clear existing loggers to ensure fresh creation
    logging.Logger.manager.loggerDict.clear()


def test_get_logger(reset_logging: None) -> None:
    """Test that get_logger returns a logger."""
    log = get_logger(__name__)
    assert log is not None
    assert hasattr(log, "info")
    assert hasattr(log, "error")
    assert hasattr(log, "warning")
    assert hasattr(log, "debug")


def test_get_logger_name(reset_logging: None) -> None:
    """Test that logger is created with the correct name."""
    test_name = "test_logger"
    logger = get_logger(test_name)

    assert logger.name == test_name


def test_get_logger_level_from_env(reset_logging: None) -> None:
    """Test that logger level is set from environment variable."""
    # Test with DEBUG level
    with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        logger = get_logger("test_logger_debug")
        assert logger.level == logging.DEBUG

    # Test with ERROR level
    with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
        logger = get_logger("test_logger_error")
        assert logger.level == logging.ERROR


def test_get_logger_default_level(reset_logging: None) -> None:
    """Test that logger uses INFO level by default."""
    # Ensure LOG_LEVEL is not set
    with patch.dict(os.environ, clear=True):
        logger = get_logger("test_logger_default")
        assert logger.level == logging.INFO


def test_get_logger_handler_setup(reset_logging: None) -> None:
    """Test that logger has a handler configured."""
    logger = get_logger("test_handler")

    # Should have one handler
    assert len(logger.handlers) == 1

    # The handler should be a StreamHandler
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_get_logger_formatter(reset_logging: None) -> None:
    """Test that logger handler has a formatter configured."""
    logger = get_logger("test_formatter")
    handler = logger.handlers[0]

    # Should have a formatter
    assert handler.formatter is not None


def test_get_logger_no_propagation(reset_logging: None) -> None:
    """Test that logger does not propagate to the root logger."""
    logger = get_logger("test_propagation")

    # Should not propagate to root
    assert logger.propagate is False


def test_get_logger_reuse(reset_logging: None) -> None:
    """Test that get_logger returns the same logger when called with the same name."""
    logger1 = get_logger("test_reuse")
    logger2 = get_logger("test_reuse")

    # Should be the same logger instance
    assert logger1 is logger2

    # Should not add another handler
    assert len(logger1.handlers) == 1
