"""Tests for utility functions."""

import os


def test_environment_variables():
    """Test that necessary environment variables are set."""
    assert "OPENAI_API_KEY" in os.environ


def test_dummy():
    """A dummy test that always passes."""
    assert 1 == 1
