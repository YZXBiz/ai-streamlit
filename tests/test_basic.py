"""Basic tests that don't require external dependencies."""

import json
import os

import pytest


def test_env_variables():
    """Test that environment variables are set correctly."""
    assert "OPENAI_API_KEY" in os.environ


def test_json_handling():
    """Test basic JSON handling."""
    data = {"name": "test", "value": 123}
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed["name"] == "test"
    assert parsed["value"] == 123


def test_pytest_working():
    """Simple test to verify pytest is working."""
    assert True


def test_basic_math():
    """Test basic arithmetic."""
    assert 1 + 1 == 2
    assert 5 * 5 == 25
    assert 10 / 2 == 5
    assert 10 % 3 == 1
