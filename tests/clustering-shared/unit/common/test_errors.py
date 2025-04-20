"""Tests for error handling utilities."""

from clustering.shared.common.errors import format_error, get_system_info


def test_format_error():
    """Test that format_error returns a formatted error message."""
    try:
        # Deliberately cause an exception
        result = 1 / 0
    except Exception as e:
        result = format_error(e)

    # Check basic structure
    assert "Error Type:" in result
    assert "ZeroDivisionError" in result
    assert "Error Message:" in result
    assert "division by zero" in result
    assert "Stack Trace:" in result


def test_get_system_info():
    """Test that get_system_info returns system information."""
    info = get_system_info()

    # Check basic structure
    assert isinstance(info, dict)
    assert "platform" in info
    assert "python_version" in info
    assert "memory" in info
