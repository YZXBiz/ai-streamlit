"""Tests for error handling utilities."""

import pytest

from clustering.shared.common.errors import format_error, get_system_info


def test_format_error():
    """Test error formatting with traceback."""
    try:
        # Deliberately cause an exception
        1 / 0
    except Exception as e:
        result = format_error(e)
        # Check that the error message contains the expected components
        assert "ZeroDivisionError" in result
        assert "division by zero" in result
        assert "Traceback" in result


def test_get_system_info():
    """Test system information retrieval."""
    sys_info = get_system_info()
    
    # Verify all the expected keys are present
    assert "platform" in sys_info
    assert "python_version" in sys_info
    assert "python_implementation" in sys_info
    assert "cpu_count" in sys_info
    assert "architecture" in sys_info
    assert "processor" in sys_info
    
    # Verify values are non-empty strings or tuples (for architecture)
    assert isinstance(sys_info["platform"], str) and sys_info["platform"]
    assert isinstance(sys_info["python_version"], str) and sys_info["python_version"]
    assert isinstance(sys_info["cpu_count"], str) and sys_info["cpu_count"]  # This is actually machine info
    assert isinstance(sys_info["architecture"], tuple)
    assert isinstance(sys_info["processor"], str) 