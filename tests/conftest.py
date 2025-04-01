"""Pytest fixtures for testing."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing.

    Returns:
        pd.DataFrame: A sample DataFrame with store features.
    """
    return pd.DataFrame(
        {
            "store_id": [f"S{i:03d}" for i in range(1, 101)],
            "revenue": [100_000 + i * 1000 for i in range(100)],
            "transactions": [5_000 + i * 50 for i in range(100)],
            "basket_size": [20 + i * 0.2 for i in range(100)],
            "customer_count": [2_500 + i * 25 for i in range(100)],
            "sales_per_sq_ft": [200 + i * 2 for i in range(100)],
            "inventory_turns": [4 + i * 0.04 for i in range(100)],
            "profit_margin": [0.15 + i * 0.001 for i in range(100)],
        }
    )


@pytest.fixture
def temp_config_file() -> Generator[Path, None, None]:
    """Create a temporary config file.

    Yields:
        Path: Path to the temporary config file.
    """
    config = {
        "job": {
            "kind": "test_job",
            "logger_service": {
                "level": "INFO",
            },
            "alerts_service": {
                "enabled": False,
            },
            "params": {
                "algorithm": "kmeans",
                "n_clusters": 3,
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
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock environment variables for testing.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
    """
    env_vars = {
        "AZURE_ACCOUNT_NAME": "testaccount",
        "AZURE_TENANT_ID": "test-tenant-id",
        "AZURE_CLIENT_ID": "test-client-id",
        "AZURE_CLIENT_SECRET": "test-client-secret",
        "ACCOUNT_URL": "https://testaccount.blob.core.windows.net",
        "CONTAINER_NAME": "test-container",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
