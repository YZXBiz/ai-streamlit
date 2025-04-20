"""Test fixtures for the pipeline package."""

from collections.abc import Iterator
from typing import Any

import dagster as dg
import polars as pl
import pytest
from dagster import ResourceDefinition

# === Resource Mocks ===


class MockConfig:
    """Mock configuration resource for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mock config with provided values.

        Args:
            **kwargs: Configuration values to be set as attributes
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


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
    """Mock writer resource for testing."""

    def __init__(self) -> None:
        """Initialize the mock writer."""
        self.written_data = {}
        self.written_count = 0
        self.path = "mock/path/writer.pkl"  # Add mock path

    def write(self, data: Any, **kwargs: Any) -> None:
        """Store data that would be written.

        Args:
            data: Data that would be written
            **kwargs: Additional arguments
        """
        self.written_data[self.written_count] = data
        self.written_count += 1


# === Pytest Fixtures ===


@pytest.fixture
def mock_execution_context() -> Iterator[dg.AssetExecutionContext]:
    """Create a mock execution context for testing Dagster assets.

    Yields:
        A mock asset execution context with test resources
    """
    mock_config = MockConfig(
        # Default test config values
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

    mock_reader = MockReader({})
    mock_writer = MockWriter()

    # Create resource definitions with all required resources
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

    with dg.build_asset_context(resources=resource_defs) as context:
        # Add setup for the renamed writer if tests rely on mock behavior
        # like checking written_count or written_data
        internal_writer = context.resources.internal_cluster_assignments
        internal_writer.written_count = 0
        internal_writer.written_data = []

        def internal_mock_write(data, **kwargs):
            internal_writer.written_count += 1
            internal_writer.written_data.append(data)

        internal_writer.write = internal_mock_write

        # Setup other writers if needed by other tests
        # Example:
        # external_writer = context.resources.output_external_data_writer
        # external_writer.written_count = 0
        # external_writer.written_data = []
        # def external_mock_write(data, **kwargs):
        #     external_writer.written_count += 1
        #     external_writer.written_data.append(data)
        # external_writer.write = external_mock_write

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
def sample_clustering_results() -> dict[str, dict[str, Any]]:
    """Create sample clustering results for testing.

    Returns:
        Dictionary of clustering results by category
    """
    return {
        "category_a": {
            "optimal_clusters": 3,
            "model": "mocked_model_object",
            "metrics": {"silhouette": 0.75, "calinski_harabasz": 120.5},
        },
        "category_b": {
            "optimal_clusters": 4,
            "model": "mocked_model_object",
            "metrics": {"silhouette": 0.68, "calinski_harabasz": 95.3},
        },
    }


@pytest.fixture
def sample_cluster_assignments() -> dict[str, pl.DataFrame]:
    """Create sample cluster assignments for testing.

    Returns:
        Dictionary of cluster assignment dataframes by category
    """
    sample_assignments = {}

    for category in ["category_a", "category_b"]:
        assignments = pl.DataFrame(
            {
                "store_id": [f"store_{i}" for i in range(10)],
                "cluster": [i % 3 for i in range(10)],
                "distance": [0.1 * i for i in range(10)],
            }
        )
        sample_assignments[category] = assignments

    return sample_assignments
