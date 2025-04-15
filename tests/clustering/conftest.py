"""Pytest fixtures for clustering tests."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import yaml
from dagster import (
    build_asset_context,
    build_op_context,
)


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture for test data directory."""
    data_dir = Path("/workspaces/testing-dagster/tests/data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Create a sample DataFrame for sales data testing.

    Returns:
        pd.DataFrame: A sample DataFrame with store sales features.
    """
    return pd.DataFrame(
        {
            "SKU_NBR": [101, 102, 103, 104, 105],
            "STORE_NBR": [1, 2, 3, 1, 2],
            "CAT_DSC": ["Category A"] * 5,
            "TOTAL_SALES": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def sample_need_state_data() -> pd.DataFrame:
    """Create a sample DataFrame for need state mapping testing.

    Returns:
        pd.DataFrame: A sample DataFrame with need state mappings.
    """
    return pd.DataFrame(
        {
            "PRODUCT_ID": [101, 102, 103, 104, 105],
            "CATEGORY": ["Category A"] * 5,
            "NEED_STATE": ["State A", "State B", "State A", "State C", "State B"],
            "CDT": ["CDT A"] * 5,
            "ATTRIBUTE_1": ["Attr 1", "Attr 2", None, "Attr 4", "Attr 5"],
            "ATTRIBUTE_2": ["Attr 1", None, "Attr 3", "Attr 4", "Attr 5"],
            "ATTRIBUTE_3": [None, "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
            "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
            "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
            "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
            "PLANOGRAM_DSC": ["PG A"] * 5,
            "PLANOGRAM_NBR": [1, 2, 3, 4, 5],
            "NEW_ITEM": [False, True, False, True, False],
            "TO_BE_DROPPED": [False, True, False, True, False],
        }
    )


@pytest.fixture
def sample_polars_sales_data(sample_sales_data) -> pl.DataFrame:
    """Convert sample sales data to polars DataFrame."""
    return pl.from_pandas(sample_sales_data)


@pytest.fixture
def sample_polars_need_state_data(sample_need_state_data) -> pl.DataFrame:
    """Convert sample need state data to polars DataFrame."""
    return pl.from_pandas(sample_need_state_data)


@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary config file for testing.

    Returns:
        Path: Path to temporary config file.
    """
    config = {
        "job": {
            "kind": "test_job",
            "logger_service": {
                "level": "INFO",
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
def temp_data_dir() -> Path:
    """Create a temporary directory for test data.

    Returns:
        Path: Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock environment variables for testing.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
    """
    env_vars = {
        "DAGSTER_HOME": "/tmp/dagster_home",
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test123;EndpointSuffix=core.windows.net",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def dagster_asset_context() -> object:
    """Create a minimal Dagster asset context for testing.

    Returns:
        A Dagster asset context.
    """
    return build_asset_context(
        resources={"io_manager": {"config": {"base_path": "/tmp/test_dagster"}}}
    )


@pytest.fixture
def dagster_op_context() -> object:
    """Create a minimal Dagster op context for testing.

    Returns:
        A Dagster op context.
    """
    return build_op_context(
        resources={"io_manager": {"config": {"base_path": "/tmp/test_dagster"}}}
    )


@pytest.fixture
def mock_csv_file(temp_data_dir: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a mock CSV file for testing.

    Args:
        temp_data_dir: Temporary directory to create file in.
        sample_sales_data: Sample data to write to CSV.

    Returns:
        Path: Path to the CSV file.
    """
    csv_path = temp_data_dir / "test_data.csv"
    sample_sales_data.to_csv(csv_path, index=False)
    return csv_path
