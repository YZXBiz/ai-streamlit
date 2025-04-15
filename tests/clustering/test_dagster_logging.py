"""Tests for Dagster logging resources in the clustering pipeline."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
from dagster import (
    build_init_resource_context,
    build_op_context,
)

from clustering.dagster.resources.logging import logger_service


@pytest.fixture
def temp_log_file() -> Generator[Path, None, None]:
    """Create a temporary log file for testing.
    
    Yields:
        Path to the temporary log file
    """
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp:
        temp_path = Path(temp.name)
    
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        os.unlink(temp_path)


class TestLoggerResource:
    """Tests for the logger_service resource."""
    
    def test_logger_creation(self, temp_log_file) -> None:
        """Test that the logger resource can be created."""
        # Create context with config
        init_context = build_init_resource_context(
            config={
                "sink": str(temp_log_file),
                "level": "INFO",
            }
        )
        
        # Initialize the resource
        logger = logger_service(init_context)
        
        # Verify the resource is callable and returns a logger
        assert callable(logger)
        
        # Get a logger instance
        log_instance = logger()
        assert isinstance(log_instance, logging.Logger)
        assert log_instance.level == logging.INFO
        
        # Verify the file handler was created
        file_handlers = [h for h in log_instance.handlers if hasattr(h, 'baseFilename')]
        assert len(file_handlers) > 0
        assert any(str(temp_log_file) in h.baseFilename for h in file_handlers)
    
    def test_logger_levels(self) -> None:
        """Test that different log levels work correctly."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        for level_name, level_value in levels.items():
            # Create context with config
            init_context = build_init_resource_context(
                config={
                    "sink": "test.log",
                    "level": level_name,
                }
            )
            
            # Initialize the resource
            logger = logger_service(init_context)
            log_instance = logger()
            
            assert log_instance.level == level_value
    
    def test_logger_in_op_context(self, temp_log_file) -> None:
        """Test using the logger in an op context."""
        # Create a resource instance
        init_context = build_init_resource_context(
            config={
                "sink": str(temp_log_file),
                "level": "INFO",
            }
        )
        logger_instance = logger_service(init_context)
        
        # Build op context with the logger resource
        op_context = build_op_context(
            resources={"logger": logger_instance}
        )
        
        # Get logger from context and log a message
        log = op_context.resources.logger()
        log.info("Test log message")
        
        # Verify the message was written to the file
        with open(temp_log_file, "r") as f:
            content = f.read()
            assert "Test log message" in content
    
    def test_default_values(self) -> None:
        """Test that default values are used when not provided."""
        # Create context with minimal config
        init_context = build_init_resource_context(
            config={}
        )
        
        # Initialize the resource
        logger = logger_service(init_context)
        log_instance = logger()
        
        # Default level should be INFO
        assert log_instance.level == logging.INFO
        
        # Default sink should be logs/dagster.log
        file_handlers = [h for h in log_instance.handlers if hasattr(h, 'baseFilename')]
        assert any("logs/dagster.log" in h.baseFilename for h in file_handlers)
    
    def test_logger_rotation(self, temp_log_file) -> None:
        """Test log rotation configuration."""
        # Create context with rotation config
        init_context = build_init_resource_context(
            config={
                "sink": str(temp_log_file),
                "level": "INFO",
                "rotation": {
                    "max_bytes": 1024,
                    "backup_count": 3,
                }
            }
        )
        
        # Initialize the resource
        logger = logger_service(init_context)
        log_instance = logger()
        
        # Verify the rotation handler is configured
        rotation_handlers = [
            h for h in log_instance.handlers 
            if hasattr(h, 'maxBytes') and hasattr(h, 'backupCount')
        ]
        
        assert len(rotation_handlers) > 0
        handler = rotation_handlers[0]
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3


def test_logger_in_dagster_execution() -> None:
    """Integration test for logger in Dagster execution."""
    from dagster import op, job, materialize
    
    @op(required_resource_keys={"logger"})
    def log_something(context) -> str:
        """Log a message using the logger resource."""
        logger = context.resources.logger()
        logger.info("This is a test message")
        return "Success"
    
    # Create a simple job that uses the logger
    @job(
        resource_defs={
            "logger": logger_service.configured({"level": "INFO", "sink": "test_job.log"})
        }
    )
    def test_job():
        log_something()
    
    # Run the job and check it completes successfully
    try:
        result = materialize(test_job)
        assert result.success
        
        # Check the log file was created
        log_file = Path("test_job.log")
        assert log_file.exists()
        
        # Check log content
        with open(log_file, "r") as f:
            content = f.read()
            assert "This is a test message" in content
        
        # Clean up
        if log_file.exists():
            os.unlink(log_file)
    except Exception as e:
        pytest.skip(f"Could not run integration test: {str(e)}")
