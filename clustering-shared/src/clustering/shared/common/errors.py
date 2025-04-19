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
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    return "".join(tb)


def get_system_info() -> dict[str, Any]:
    """Get system information for error reporting.

    Returns:
        Dictionary with system information
    """
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "cpu_count": platform.machine(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
    }
