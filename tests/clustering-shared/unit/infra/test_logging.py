"""Tests for the logging service module."""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from clustering.shared.infra.logging import LoggerService


class TestLoggerService:
    """Tests for the LoggerService class."""

    def test_default_initialization(self) -> None:
        """Test that LoggerService initializes with correct defaults."""
        service = LoggerService()
        assert service.sink == "logs/app.log"
        assert service.level == "DEBUG"
        assert service.format == "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {message}"
        assert service.colorize is True
        assert service.serialize is False
        assert service.backtrace is True
        assert service.diagnose is False
        assert service.catch is True

    def test_custom_initialization(self) -> None:
        """Test that LoggerService initializes with custom values."""
        service = LoggerService(
            sink="custom.log",
            level="ERROR",
            colorize=False,
            serialize=True,
        )
        assert service.sink == "custom.log"
        assert service.level == "ERROR"
        assert service.colorize is False
        assert service.serialize is True

    @patch("loguru.logger.remove")
    @patch("loguru.logger.add")
    @patch("pathlib.Path.mkdir")
    def test_start(self, mock_mkdir: MagicMock, mock_add: MagicMock, mock_remove: MagicMock) -> None:
        """Test starting the logging service."""
        service = LoggerService(sink="logs/custom.log")
        
        # Mock datetime to get a consistent timestamp
        with patch("clustering.shared.infra.logging.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20230101_120000"
            
            service.start()
            
            # Verify logger.remove was called
            mock_remove.assert_called_once()
            
            # Verify logs directory creation
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            # Verify logger.add was called with expected parameters
            args, kwargs = mock_add.call_args
            assert kwargs["sink"] == "logs/custom_20230101_120000.log"
            assert kwargs["level"] == "DEBUG"
            assert kwargs["colorize"] is True

    def test_logger(self) -> None:
        """Test retrieving the logger instance."""
        service = LoggerService()
        logger = service.logger()
        
        # Check that we got the loguru logger
        assert logger.__module__ == "loguru._logger"
        
    @patch("loguru.logger.remove")
    @patch("loguru.logger.add")
    @patch("pathlib.Path.mkdir")
    def test_stop(self, mock_mkdir: MagicMock, mock_add: MagicMock, mock_remove: MagicMock) -> None:
        """Test stopping the logging service."""
        service = LoggerService()
        service.stop()
        # Nothing to verify since the stop method is a no-op,
        # but we're testing for code coverage 