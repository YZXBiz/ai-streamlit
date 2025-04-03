"""Integration tests for job workflows."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pandas as pd
import pytest
import yaml
from clustering import settings
from clustering.io import config_parser


@pytest.fixture
def sample_workflow_config() -> Dict[str, Any]:
    """Create a sample workflow configuration.

    Returns:
        Dict[str, Any]: A sample workflow configuration.
    """
    return {
        "job": {
            "kind": "internal_preprocessing",
            "logger_service": {
                "level": "INFO",
                "format": "[{time:YYYY-MM-DD HH:mm:ss}] {level} | {message}",
            },
            "alerts_service": {
                "enabled": False,
            },
            "source": {
                "type": "dataframe",  # Use in-memory dataframe for testing
            },
            "params": {
                "features_to_include": [
                    "store_id",
                    "revenue",
                    "transactions",
                ],
                "outlier_removal": {
                    "method": "iqr",
                    "threshold": 1.5,
                },
                "imputation": {
                    "method": "mean",
                },
                "scaling": {
                    "method": "standard",
                },
            },
            "output": {
                "type": "memory",  # Keep result in memory for testing
            },
        }
    }


@pytest.fixture
def sample_workflow_config_file(sample_workflow_config: Dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary workflow config file.

    Args:
        sample_workflow_config: The configuration to write to the file.

    Yields:
        Path: Path to the temporary config file.
    """
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp:
        temp_path = Path(temp.name)
        yaml.dump(sample_workflow_config, temp)

    yield temp_path

    if temp_path.exists():
        os.unlink(temp_path)


def test_preprocessing_workflow(
    sample_data: pd.DataFrame, sample_workflow_config: Dict[str, Any], mock_env_vars: None
) -> None:
    """Test the preprocessing workflow end-to-end.

    Args:
        sample_data: Sample DataFrame for testing.
        sample_workflow_config: Sample workflow configuration.
        mock_env_vars: Mock environment variables.
    """
    # Inject the sample data
    sample_workflow_config["job"]["source"]["data"] = sample_data

    # Convert config to settings object
    config_obj = config_parser.to_object(sample_workflow_config)
    setting = settings.MainSettings.model_validate(config_obj)

    # Run the preprocessing job
    with setting.job as runner:
        result = runner.run()

    # Check that the result is a dictionary
    assert isinstance(result, dict)

    # Check that the preprocessed data is in the result
    assert "preprocessed_data" in result

    # Check that the preprocessed data has the expected shape
    preprocessed_data = result["preprocessed_data"]
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert not preprocessed_data.empty

    # Check that the features were properly selected
    expected_columns = ["store_id", "revenue", "transactions"]
    assert all(col in preprocessed_data.columns for col in expected_columns)

    # Check that scaling was applied (values should be centered around 0)
    for col in ["revenue", "transactions"]:
        assert abs(preprocessed_data[col].mean()) < 0.1  # Close to 0 after scaling


def test_end_to_end_workflow(sample_data: pd.DataFrame, mock_env_vars: None) -> None:
    """Test the entire workflow from preprocessing to clustering.

    Args:
        sample_data: Sample DataFrame for testing.
        mock_env_vars: Mock environment variables.
    """
    # Step 1: Preprocessing
    preprocessing_config = {
        "job": {
            "kind": "internal_preprocessing",
            "logger_service": {"level": "INFO"},
            "alerts_service": {"enabled": False},
            "source": {
                "type": "dataframe",
                "data": sample_data,
            },
            "params": {
                "features_to_include": [
                    "store_id",
                    "revenue",
                    "transactions",
                ],
                "outlier_removal": {"method": "iqr", "threshold": 1.5},
                "imputation": {"method": "mean"},
                "scaling": {"method": "standard"},
            },
            "output": {"type": "memory"},
        }
    }

    # Convert config to settings object
    config_obj = config_parser.to_object(preprocessing_config)
    setting = settings.MainSettings.model_validate(config_obj)

    # Run preprocessing
    with setting.job as runner:
        preprocessing_result = runner.run()

    preprocessed_data = preprocessing_result["preprocessed_data"]

    # Step 2: Clustering
    clustering_config = {
        "job": {
            "kind": "internal_clustering",
            "logger_service": {"level": "INFO"},
            "alerts_service": {"enabled": False},
            "source": {
                "type": "dataframe",
                "data": preprocessed_data,
            },
            "params": {
                "algorithm": "kmeans",
                "n_clusters": 3,
                "random_state": 42,
            },
            "output": {"type": "memory"},
        }
    }

    # Convert config to settings object
    config_obj = config_parser.to_object(clustering_config)
    setting = settings.MainSettings.model_validate(config_obj)

    # Run clustering
    with setting.job as runner:
        clustering_result = runner.run()

    # Check that the result is a dictionary
    assert isinstance(clustering_result, dict)

    # Check that the cluster assignments are in the result
    assert "clustered_data" in clustering_result

    # Check that the cluster assignments have the expected shape
    clustered_data = clustering_result["clustered_data"]
    assert isinstance(clustered_data, pd.DataFrame)
    assert not clustered_data.empty

    # Check that cluster labels were assigned
    assert "cluster" in clustered_data.columns

    # Check that we have the expected number of clusters
    assert len(clustered_data["cluster"].unique()) == 3
