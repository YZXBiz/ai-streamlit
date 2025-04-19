"""Pytest fixtures for testing the clustering-cli package."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner.

    Returns:
        CliRunner: A Click CLI test runner.
    """
    return CliRunner()


@pytest.fixture
def mock_config_file() -> Generator[Path, None, None]:
    """Create a mock configuration file for CLI testing.

    Yields:
        Path: Path to the temporary configuration file.
    """
    config = {
        "job": {
            "kind": "test_job",
            "logger_service": {
                "level": "INFO",
            },
            "params": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 42,
            },
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp:
        temp_path = Path(temp.name)
        yaml.dump(config, temp)

    yield temp_path

    if temp_path.exists():
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for CLI output files.

    Yields:
        Path: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path
