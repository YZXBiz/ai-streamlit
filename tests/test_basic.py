"""
Basic tests to verify that the testing infrastructure is working.
"""

from typing import Any


def test_basic_pass() -> None:
    """Simple test that always passes."""
    assert True is True


def test_basic_arithmetic() -> None:
    """Test basic arithmetic operations."""
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
    assert 10 / 2 == 5.0


def test_string_operations() -> None:
    """Test basic string operations."""
    assert "hello" + " world" == "hello world"
    assert "hello".upper() == "HELLO"
    assert "WORLD".lower() == "world"
    assert "hello world".split() == ["hello", "world"]
