"""Test fixtures for the pipeline package."""

from collections.abc import Iterator
from typing import Any, Dict
from unittest.mock import MagicMock

import dagster as dg
import polars as pl
import pytest
from dagster import ResourceDefinition, build_asset_context, build_op_context

# === Resource Mocks ===


class MockConfig:
    """Mock configuration resource for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mock config with provided values.

        Args:
            **kwargs: Configuration values to be set as attributes
        """
        self._attributes = kwargs

    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_attributes":
            super().__setattr__(name, value)
        else:
            self._attributes[name] = value


class MockReader:
    """Mock reader resource for testing."""

    def __init__(self, data: Any) -> None:
        """Initialize with predefined data.

        Args:
            data: Data to return when read is called
        """
        self.data = data
        self.path = "mock/path/reader.pkl"  # Add mock path

    def read(self) -> Any:
        """Return predefined data.

        Returns:
            The data provided during initialization
        """
        return self.data


class MockWriter:
    """Mock writer for testing."""

    def __init__(self):
        """Initialize the mock writer."""
        self.write = MagicMock(return_value="/mock/path/output.parquet")
        self.written_data = []  # Keep track of written data
        self.written_count = 0  # Count of write operations


# === Pytest Fixtures ===


@pytest.fixture
def mock_execution_context() -> Iterator[dg.AssetExecutionContext]:
    """Create a mock execution context for testing Dagster assets.

    Yields:
        A mock asset execution context with test resources
    """
    # Default test config values
    mock_config = MockConfig(
        ignore_features=[],
        imputation_type="simple",
        numeric_imputation="mean",
        categorical_imputation="mode",
        normalize=True,
        norm_method="robust",
        outlier_detection=True,
        outlier_threshold=0.05,
        outlier_method="iforest",
        pca_active=True,
        pca_components=0.8,
        pca_method="linear",
        metadata_detail="full",
        remove_duplicates=True,
        remove_null_threshold=0.5,
        standardize_features=True,
        normalize_method="min_max",
    )

    # Configure mock readers and writers
    mock_reader = MockReader({})
    mock_writer = MockWriter()

    # Create resource definitions
    resource_defs = {
        "config": ResourceDefinition.hardcoded_resource(mock_config),
        # External data resources
        "input_external_placerai_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "input_external_urbanicity_template_reader": ResourceDefinition.hardcoded_resource(
            mock_reader
        ),
        "input_external_urbanicity_experiment_reader": ResourceDefinition.hardcoded_resource(
            mock_reader
        ),
        "external_features_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "output_external_data_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        # Internal data resources
        "sales_data_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "category_mapping_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "sales_by_category_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "sales_by_category_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        # Clustering resources
        "internal_cluster_assignments": ResourceDefinition.hardcoded_resource(mock_writer),
        "clustering_models_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        "merged_cluster_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        "internal_model_output": ResourceDefinition.hardcoded_resource(mock_writer),
        "external_cluster_assignments": ResourceDefinition.hardcoded_resource(mock_reader),
        "external_model_output": ResourceDefinition.hardcoded_resource(mock_reader),
        "job_params": ResourceDefinition.hardcoded_resource(MockConfig(min_cluster_size=10)),
        "merged_cluster_assignments": ResourceDefinition.hardcoded_resource(mock_writer),
    }

    # Use Dagster's built-in op context
    with build_op_context(resources=resource_defs) as context:
        yield context


@pytest.fixture
def sample_category_data() -> dict[str, pl.DataFrame]:
    """Create sample category data for testing.

    Returns:
        Dictionary of sample dataframes by category
    """
    categories = ["category_a", "category_b"]
    sample_data = {}

    for category in categories:
        # Create sample data with some missing values and typical features
        df = pl.DataFrame(
            {
                "store_id": [f"store_{i}" for i in range(10)],
                "feature_1": [i * 10 for i in range(10)],
                "feature_2": [i * i for i in range(10)],
                "feature_3": [None if i % 3 == 0 else i * 5 for i in range(10)],
                "feature_4": ["A" if i % 2 == 0 else "B" for i in range(10)],
            }
        )
        sample_data[category] = df

    return sample_data


@pytest.fixture
def sample_clustering_results() -> dict[str, int]:
    """Create sample optimal cluster counts for testing.

    Returns:
        Dictionary mapping category names to optimal cluster counts (integers)
    """
    return {"category_a": 3, "category_b": 4}


@pytest.fixture
def sample_trained_models() -> dict[str, dict[str, Any]]:
    """Create sample trained models data for testing.

    Returns:
        Dictionary of trained model information by category
    """
    return {
        "category_a": {
            "num_clusters": 3,
            "model": "mocked_model_object",
            "metrics": {"silhouette": 0.75, "calinski_harabasz": 120.5},
            "num_samples": 100,
            "features": ["feature_1", "feature_2", "feature_3"],
            "experiment_path": "mock/path/experiments/category_a",
        },
        "category_b": {
            "num_clusters": 4,
            "model": "mocked_model_object",
            "metrics": {"silhouette": 0.68, "calinski_harabasz": 95.3},
            "num_samples": 120,
            "features": ["feature_1", "feature_2", "feature_3"],
            "experiment_path": "mock/path/experiments/category_b",
        },
    }


@pytest.fixture
def mock_writer() -> MockWriter:
    """Create a mock writer for testing.

    Returns:
        A MockWriter instance.
    """
    return MockWriter()
