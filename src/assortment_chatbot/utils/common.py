"""
Common utility functions for the Assortment Chatbot application.

This module contains reusable utility functions that are
used across multiple parts of the application.
"""

from pathlib import Path


def find_project_root() -> Path:
    """
    Find the project root by looking for pyproject.toml file.

    Returns:
        Path: The project root directory path
    """
    path = Path(__file__).resolve().parent

    # Look up to 4 levels up for pyproject.toml
    for _ in range(5):
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent

    # If not found, return a reasonable default (3 directories up from utils)
    return Path(__file__).resolve().parent.parent.parent.parent
