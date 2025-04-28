"""Tests for repository module."""

import pytest

def test_repo_constants():
    """Test repository module constants."""
    from app.ports.repository import Repository
    assert hasattr(Repository, "__abstractmethods__")
