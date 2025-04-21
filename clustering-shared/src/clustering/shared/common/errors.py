"""Error handling utilities."""

import platform
import sys
import traceback
from typing import Any


def format_error(error: Exception) -> str:
    """Format an exception with traceback.

    Args:
        error: The exception to format

    Returns:
        Formatted error message with traceback
    """
    tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))

    # Create structured error message
    error_type = type(error).__name__
    error_message = str(error)

    structured_error = (
        f"Error Type: {error_type}\nError Message: {error_message}\nStack Trace:\n{tb_str}"
    )

    return structured_error


def get_system_info() -> dict[str, Any]:
    """Get system information for error reporting.

    Returns:
        Dictionary with system information
    """
    # Get memory information
    try:
        import psutil

        memory_info = psutil.virtual_memory()
        memory = {
            "total": memory_info.total,
            "available": memory_info.available,
            "percent_used": memory_info.percent,
        }
    except ImportError:
        memory = {"error": "psutil not available"}

    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "cpu_count": platform.machine(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "memory": memory,
    }
