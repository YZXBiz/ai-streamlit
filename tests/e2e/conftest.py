"""Configuration for end-to-end tests.

This module provides common fixtures and configuration for e2e tests.
"""

import os
import sys
import pytest
from pathlib import Path
from collections.abc import Generator


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    # Assume the project root is the parent of the tests directory
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def add_project_to_path(project_root: Path) -> Generator[None, None, None]:
    """Add the project root to the Python path."""
    sys.path.insert(0, str(project_root))
    yield
    sys.path.remove(str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_env(add_project_to_path: None) -> Generator[None, None, None]:
    """Set up the testing environment."""
    # Save original environment variables
    original_env = os.environ.copy()
    
    # Set testing environment variables
    os.environ["ENV"] = "test"
    os.environ["TEST_MODE"] = "true"
    
    # Ensure logs are set to a test-friendly level
    os.environ["LOG_LEVEL"] = "ERROR"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with standard directories."""
    # Create standard directories
    data_dir = tmp_path / "data"
    (data_dir / "internal").mkdir(parents=True)
    (data_dir / "external").mkdir(parents=True)
    (data_dir / "output").mkdir(parents=True)
    
    # Set data directory in environment
    os.environ["DATA_DIR"] = str(data_dir)
    
    return tmp_path 