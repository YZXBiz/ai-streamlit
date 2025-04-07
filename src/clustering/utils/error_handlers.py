"""Error handling chain implementation using the Chain of Responsibility pattern."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

# Type for error classes
E = TypeVar("E", bound=Exception)


class ErrorContext:
    """Context information for error handling."""

    def __init__(
        self,
        operation: str = "",
        source: str = "",
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize error context.

        Args:
            operation: The operation being performed when the error occurred
            source: The source of the operation (e.g., file name, component name)
            data: Data related to the operation
            metadata: Additional metadata about the error context
        """
        self.operation = operation
        self.source = source
        self.data = data
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """String representation of the error context."""
        parts = [f"Operation: {self.operation}"]
        if self.source:
            parts.append(f"Source: {self.source}")
        if self.metadata:
            parts.append(f"Metadata: {self.metadata}")
        return ", ".join(parts)


class ErrorHandler(ABC):
    """Base class for error handlers in the chain of responsibility."""

    def __init__(self, successor: Optional["ErrorHandler"] = None):
        """Initialize the error handler.

        Args:
            successor: The next handler in the chain
        """
        self._successor = successor

    def set_successor(self, successor: "ErrorHandler") -> None:
        """Set the successor for this handler.

        Args:
            successor: The next handler in the chain
        """
        self._successor = successor

    def handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle the error or pass it to the successor.

        Args:
            error: The exception to handle
            context: Context information for error handling

        Raises:
            Exception: If the error cannot be handled by any handler in the chain
        """
        if self._can_handle(error, context):
            return self._do_handle(error, context)
        elif self._successor:
            return self._successor.handle_error(error, context)
        else:
            # If no handler in the chain can handle the error, raise it
            raise error

    @abstractmethod
    def _can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle the given error.

        Args:
            error: The exception to check
            context: Context information for error handling

        Returns:
            True if this handler can handle the error, False otherwise
        """
        pass

    @abstractmethod
    def _do_handle(self, error: Exception, context: ErrorContext) -> None:
        """Handle the error.

        Args:
            error: The exception to handle
            context: Context information for error handling
        """
        pass


class TypeBasedErrorHandler(ErrorHandler):
    """Error handler that handles errors based on their type."""

    def __init__(
        self,
        error_types: list[type[Exception]],
        successor: ErrorHandler | None = None,
    ):
        """Initialize the type-based error handler.

        Args:
            error_types: List of exception types this handler can handle
            successor: The next handler in the chain
        """
        super().__init__(successor)
        self.error_types = error_types

    def _can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle the given error based on its type.

        Args:
            error: The exception to check
            context: Context information for error handling

        Returns:
            True if the error type is in the list of handled types, False otherwise
        """
        return any(isinstance(error, error_type) for error_type in self.error_types)


class LoggingErrorHandler(TypeBasedErrorHandler):
    """Error handler that logs errors."""

    def __init__(
        self,
        error_types: list[type[Exception]],
        logger: logging.Logger | None = None,
        log_level: int = logging.ERROR,
        successor: ErrorHandler | None = None,
    ):
        """Initialize the logging error handler.

        Args:
            error_types: List of exception types this handler can handle
            logger: Logger to use (if None, uses the root logger)
            log_level: Logging level to use
            successor: The next handler in the chain
        """
        super().__init__(error_types, successor)
        self.logger = logger or logging.getLogger()
        self.log_level = log_level

    def _do_handle(self, error: Exception, context: ErrorContext) -> None:
        """Log the error.

        Args:
            error: The exception to handle
            context: Context information for error handling
        """
        error_type = type(error).__name__
        message = f"{error_type} in {context}: {str(error)}"
        self.logger.log(self.log_level, message, exc_info=True)


class RetryErrorHandler(TypeBasedErrorHandler):
    """Error handler that retries operations on certain errors."""

    def __init__(
        self,
        error_types: list[type[Exception]],
        max_retries: int = 3,
        retry_callback: callable | None = None,
        successor: ErrorHandler | None = None,
    ):
        """Initialize the retry error handler.

        Args:
            error_types: List of exception types this handler can handle
            max_retries: Maximum number of retries
            retry_callback: Callback function to execute the operation again
            successor: The next handler in the chain
        """
        super().__init__(error_types, successor)
        self.max_retries = max_retries
        self.retry_callback = retry_callback
        self._retry_counts: dict[str, int] = {}

    def _get_operation_key(self, context: ErrorContext) -> str:
        """Get a unique key for the operation to track retry counts.

        Args:
            context: Context information for error handling

        Returns:
            A string key representing the operation
        """
        return f"{context.operation}:{context.source}"

    def _do_handle(self, error: Exception, context: ErrorContext) -> None:
        """Retry the operation if the retry count hasn't been exceeded.

        Args:
            error: The exception to handle
            context: Context information for error handling

        Raises:
            Exception: If the retry count is exceeded or no retry callback is set
        """
        if not self.retry_callback:
            # Can't retry without a callback, pass to successor
            if self._successor:
                return self._successor.handle_error(error, context)
            else:
                raise error

        operation_key = self._get_operation_key(context)
        retry_count = self._retry_counts.get(operation_key, 0)

        if retry_count < self.max_retries:
            # Increment retry count
            self._retry_counts[operation_key] = retry_count + 1

            # Log the retry
            logging.info(
                f"Retrying {context.operation} (attempt {retry_count + 1}/{self.max_retries}) "
                f"after error: {str(error)}"
            )

            # Execute the retry callback
            return self.retry_callback(context)
        else:
            # Max retries exceeded, pass to successor
            logging.warning(
                f"Max retries ({self.max_retries}) exceeded for {context.operation}: {str(error)}"
            )
            if self._successor:
                return self._successor.handle_error(error, context)
            else:
                raise error


class FallbackErrorHandler(TypeBasedErrorHandler):
    """Error handler that provides fallback behavior for certain errors."""

    def __init__(
        self,
        error_types: list[type[Exception]],
        fallback_value: Any,
        successor: ErrorHandler | None = None,
    ):
        """Initialize the fallback error handler.

        Args:
            error_types: List of exception types this handler can handle
            fallback_value: Value to return when an error occurs
            successor: The next handler in the chain
        """
        super().__init__(error_types, successor)
        self.fallback_value = fallback_value

    def _do_handle(self, error: Exception, context: ErrorContext) -> Any:
        """Return the fallback value.

        Args:
            error: The exception to handle
            context: Context information for error handling

        Returns:
            The fallback value
        """
        logging.info(f"Using fallback value for {context.operation} after error: {str(error)}")
        return self.fallback_value


class ErrorHandlerChain:
    """Factory class for creating common error handler chains."""

    @staticmethod
    def create_default_chain() -> ErrorHandler:
        """Create a default error handler chain.

        Returns:
            The head of the error handler chain
        """
        # Create handlers
        logging_handler = LoggingErrorHandler([Exception])

        # Set up the chain
        # In this simple case, there's only one handler
        return logging_handler

    @staticmethod
    def create_io_chain(retry_callback: callable | None = None) -> ErrorHandler:
        """Create an error handler chain for IO operations.

        Args:
            retry_callback: Callback function to execute the operation again

        Returns:
            The head of the error handler chain
        """
        # Create handlers
        io_errors = [IOError, FileNotFoundError, PermissionError]
        retry_handler = RetryErrorHandler(io_errors, retry_callback=retry_callback)
        logging_handler = LoggingErrorHandler([Exception])

        # Set up the chain
        retry_handler.set_successor(logging_handler)

        return retry_handler

    @staticmethod
    def create_data_processing_chain(fallback_value: Any = None) -> ErrorHandler:
        """Create an error handler chain for data processing operations.

        Args:
            fallback_value: Value to return when an error occurs

        Returns:
            The head of the error handler chain
        """
        # Create handlers
        data_errors = [ValueError, TypeError, IndexError, KeyError]
        fallback_handler = FallbackErrorHandler(data_errors, fallback_value)
        logging_handler = LoggingErrorHandler([Exception])

        # Set up the chain
        fallback_handler.set_successor(logging_handler)

        return fallback_handler
