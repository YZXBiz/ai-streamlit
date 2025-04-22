"""
Logging utility for the application.

This module provides a centralized logging configuration with MDC (Mapped Diagnostic Context)
capabilities using Python's standard logging module and a thread-local context.
"""

import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# Thread-local storage for MDC
_mdc_context = threading.local()

# Default MDC values
DEFAULT_MDC = {
    "app_name": "data-chat-assistant",
    "version": "0.1.0",
}


def init_mdc() -> None:
    """Initialize the MDC context for the current thread if not already initialized."""
    if not hasattr(_mdc_context, "context"):
        _mdc_context.context = DEFAULT_MDC.copy()


def get_mdc() -> Dict[str, Any]:
    """Get the current MDC context dictionary.
    
    Returns:
        A dictionary containing all MDC values for the current thread.
    """
    init_mdc()
    return _mdc_context.context


def put_mdc(key: str, value: Any) -> None:
    """Add a key-value pair to the MDC context.
    
    Args:
        key: The MDC key
        value: The value to store
    """
    init_mdc()
    _mdc_context.context[key] = value


def remove_mdc(key: str) -> None:
    """Remove a key from the MDC context.
    
    Args:
        key: The MDC key to remove
    """
    init_mdc()
    if key in _mdc_context.context:
        del _mdc_context.context[key]


def clear_mdc() -> None:
    """Clear all MDC values and reset to defaults."""
    if hasattr(_mdc_context, "context"):
        _mdc_context.context = DEFAULT_MDC.copy()


@contextmanager
def mdc_context(**kwargs) -> None:
    """Context manager for temporarily setting MDC values.
    
    Args:
        **kwargs: Key-value pairs to add to the MDC context
        
    Example:
        ```python
        with mdc_context(user_id="user123", request_id=uuid.uuid4().hex):
            logger.info("Processing user request")
        ```
    """
    init_mdc()
    original = _mdc_context.context.copy()
    
    try:
        for key, value in kwargs.items():
            put_mdc(key, value)
        yield
    finally:
        _mdc_context.context = original


class MDCFilter(logging.Filter):
    """Filter that adds MDC values to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add MDC context to the log record.
        
        Args:
            record: The log record to process
            
        Returns:
            True (always passes the filter)
        """
        mdc = get_mdc()
        for key, value in mdc.items():
            setattr(record, key, value)
        
        # Add request_id if not present
        if "request_id" not in mdc:
            setattr(record, "request_id", "-")
            
        return True


class MDCAdapter(logging.LoggerAdapter):
    """Adapter that adds MDC context to log messages."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the log message and kwargs to include MDC context.
        
        Args:
            msg: The log message
            kwargs: Additional logging parameters
            
        Returns:
            Tuple of (processed_message, kwargs)
        """
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured with MDC capabilities.
    
    Args:
        name: The name of the logger, typically __name__
        
    Returns:
        A configured logger with MDC support
    """
    logger = logging.getLogger(name)
    
    # Add MDC filter if not already added
    has_mdc_filter = any(isinstance(f, MDCFilter) for f in logger.filters)
    if not has_mdc_filter:
        logger.addFilter(MDCFilter())
    
    return MDCAdapter(logger, {})


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> None:
    """Configure the root logger with console and/or file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file (if None, file logging is disabled)
        console: Whether to enable console logging
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter with MDC values
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [rid:%(request_id)s] - %(message)s"
    )
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    log_success: bool = True,
    log_exception: bool = True,
) -> None:
    """Context manager for logging operations with timing information.
    
    Args:
        logger: The logger to use
        operation: The name of the operation being performed
        log_success: Whether to log successful operations
        log_exception: Whether to log exceptions
        
    Example:
        ```python
        with log_operation(logger, "database_query"):
            results = db.execute_query(sql)
        ```
    """
    start_time = time.time()
    request_id = uuid.uuid4().hex
    
    with mdc_context(request_id=request_id):
        logger.info(f"Starting operation: {operation}")
        
        try:
            yield
            if log_success:
                duration = time.time() - start_time
                logger.info(f"Operation {operation} completed successfully in {duration:.2f}s")
        except Exception as e:
            if log_exception:
                duration = time.time() - start_time
                logger.exception(f"Operation {operation} failed after {duration:.2f}s: {str(e)}")
            raise 