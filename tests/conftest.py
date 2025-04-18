"""Project-wide pytest fixtures."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
import yaml


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up the test environment with common configurations.
    
    This fixture runs automatically for all tests.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
    """
    # Set common environment variables for testing
    test_env_vars = {
        "ENV": "test",
        "DAGSTER_HOME": str(Path(tempfile.gettempdir()) / "dagster_home_test"),
        "DATA_DIR": str(Path(tempfile.gettempdir()) / "test_data"),
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    
    # Create necessary directories
    for directory in [test_env_vars["DAGSTER_HOME"], test_env_vars["DATA_DIR"]]:
        Path(directory).mkdir(exist_ok=True, parents=True)


@pytest.fixture
def common_test_data() -> dict[str, Any]:
    """Provide common test data shared across all test packages.

    Returns:
        dict: Dictionary containing common test data.
    """
    return {
        "store_ids": [f"STORE_{i}" for i in range(1, 11)],
        "categories": ["Health", "Beauty", "Grocery", "Pharmacy", "General"],
        "test_dates": ["2023-01-01", "2023-01-02", "2023-01-03"],
    }


@pytest.fixture
def package_root_dir() -> Path:
    """Get the root directory of the project.
    
    Returns:
        Path: Path to the project root directory.
    """
    return Path(__file__).parent.parent


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
def sample_polars_data() -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing.

    Returns:
        pl.DataFrame: A sample DataFrame with store features.
    """
    return pl.DataFrame(
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


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file for testing.

    Yields:
        Path: Path to the temporary CSV file.
    """
    data = pd.DataFrame({
        "SKU_NBR": [1001, 1002, 1003],
        "STORE_NBR": [501, 502, 503],
        "CAT_DSC": ["Health", "Beauty", "Grocery"],
        "TOTAL_SALES": [1500.50, 2200.75, 3100.25]
    })
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
        data.to_csv(temp_path, index=False)
    
    yield temp_path
    
    if temp_path.exists():
        os.unlink(temp_path)
