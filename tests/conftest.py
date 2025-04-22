"""Test fixtures for the dashboard application."""

import os
import pathlib
from collections.abc import Iterator

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest


@pytest.fixture
def data_dir(request: FixtureRequest) -> pathlib.Path:
    """Return a Path object to the data directory.

    Args:
        request: The pytest request object

    Returns:
        Path to the data directory
    """
    return pathlib.Path(request.config.rootdir) / "data"


@pytest.fixture
def sample_cluster_data() -> pd.DataFrame:
    """Create a sample DataFrame with cluster data.

    Returns:
        DataFrame with sample cluster data
    """
    return pd.DataFrame(
        {
            "STORE_NBR": ["123", "456", "789", "101", "112"],
            "SALES": [100000, 200000, 150000, 300000, 250000],
            "TRANSACTIONS": [1000, 2000, 1500, 3000, 2500],
            "CLUSTER": [0, 1, 0, 2, 1],
            "FEATURE_1": [10, 20, 15, 30, 25],
            "FEATURE_2": [5, 10, 7, 15, 12],
        }
    )


@pytest.fixture
def sample_store_metadata() -> pd.DataFrame:
    """Create a sample DataFrame with store metadata.

    Returns:
        DataFrame with sample store metadata
    """
    return pd.DataFrame(
        {
            "STORE_NBR": ["123", "456", "789", "101", "112"],
            "REGION": ["East", "West", "East", "Central", "West"],
            "FORMAT": ["A", "B", "A", "C", "B"],
            "SQFT": [5000, 8000, 4500, 10000, 7500],
        }
    )


@pytest.fixture
def sample_cluster_features() -> pd.DataFrame:
    """Create a sample DataFrame with cluster features.

    Returns:
        DataFrame with sample cluster features
    """
    return pd.DataFrame(
        {
            "CLUSTER": [0, 1, 2],
            "FEATURE_1_MEAN": [12.5, 22.5, 30.0],
            "FEATURE_2_MEAN": [6.0, 11.0, 15.0],
            "COUNT": [2, 2, 1],
        }
    )
