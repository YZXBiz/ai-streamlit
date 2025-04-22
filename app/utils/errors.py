"""
Error handling utilities for the application.

This module provides custom exceptions and error handling utilities
to standardize error reporting and handling across the application.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

from app.utils.logging import get_logger

# Logger for error module
logger = get_logger(__name__)

# Type variables for function signatures
T = TypeVar("T")
R = TypeVar("R")


class AppError(Exception):
    """Base exception class for all application errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the application error.
        
        Args:
            message: Human-readable error message
            code: Error code for programmatic reference
            status_code: HTTP status code (for API errors)
            details: Additional error details and context
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation.
        
        Returns:
            Dict with error details
        """
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status_code": self.status_code,
                "details": self.details,
            }
        }


class ValidationError(AppError):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the validation error.
        
        Args:
            message: Human-readable error message
            details: Additional error details, like field-specific errors
        """
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class DatabaseError(AppError):
    """Exception raised for database-related errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the database error.
        
        Args:
            message: Human-readable error message
            details: Additional database-specific error details
        """
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


class NotFoundError(AppError):
    """Exception raised when a requested resource is not found."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Union[str, int],
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the not found error.
        
        Args:
            resource_type: The type of resource not found (e.g., "user", "file")
            resource_id: The identifier of the resource
            details: Additional error details
        """
        message = f"{resource_type.capitalize()} with id '{resource_id}' not found"
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404,
            details=details,
        )


class ExternalAPIError(AppError):
    """Exception raised for errors from external API calls."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the external API error.
        
        Args:
            message: Human-readable error message
            service_name: Name of the external service (e.g., "Snowflake", "OpenAI")
            details: Additional error details from the external service
        """
        super().__init__(
            message=message,
            code=f"{service_name.upper()}_API_ERROR",
            status_code=502,  # Bad Gateway
            details=details,
        )


class AuthenticationError(AppError):
    """Exception raised for authentication failures."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the authentication error.
        
        Args:
            message: Human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class AuthorizationError(AppError):
    """Exception raised for authorization failures."""
    
    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the authorization error.
        
        Args:
            message: Human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
        )


class InvalidConfigurationError(AppError):
    """Exception raised for invalid configuration."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the configuration error.
        
        Args:
            message: Human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="INVALID_CONFIGURATION",
            status_code=500,
            details=details,
        )


def handle_errors(
    error_map: Dict[Type[Exception], Callable[[Exception], Any]],
    default_handler: Optional[Callable[[Exception], Any]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator for handling exceptions according to a mapping.
    
    Args:
        error_map: Mapping from exception types to handler functions
        default_handler: Default handler for unhandled exceptions
    
    Returns:
        A decorator function
    
    Example:
        ```python
        @handle_errors({
            ValueError: lambda e: f"Invalid value: {str(e)}",
            DatabaseError: lambda e: f"Database error: {e.message}",
        })
        def process_data(data):
            # processing logic that might raise exceptions
            pass
        ```
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        def wrapper(*args: Any, **kwargs: Any) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                for exc_type, handler in error_map.items():
                    if isinstance(e, exc_type):
                        logger.error(f"Error handled by decorator: {str(e)}")
                        return cast(R, handler(e))
                
                if default_handler:
                    logger.error(f"Error handled by default handler: {str(e)}")
                    return cast(R, default_handler(e))
                
                logger.exception(f"Unhandled error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def safe_operation(func: Callable[..., T]) -> Callable[..., Union[T, None]]:
    """Decorator to safely execute a function and return None on error.
    
    Args:
        func: The function to wrap
    
    Returns:
        A wrapped function that returns None on error
    
    Example:
        ```python
        @safe_operation
        def parse_json(data: str) -> dict:
            return json.loads(data)
        ```
    """
    def wrapper(*args: Any, **kwargs: Any) -> Union[T, None]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    
    return wrapper 