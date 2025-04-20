"""Pytest configuration for clustering-pipeline tests.

This file contains fixtures and configuration for testing Dagster assets in the clustering-pipeline.
"""

import os
from typing import Any, Iterator

import dagster as dg
import pandas as pd
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
    )
    
    mock_reader = MockReader({})
    mock_writer = MockWriter()
    
    resource_defs = {
        "config": ResourceDefinition.hardcoded_resource(mock_config),
        "sales_by_category_reader": ResourceDefinition.hardcoded_resource(mock_reader),
        "cluster_assignments_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        "clustering_models_writer": ResourceDefinition.hardcoded_resource(mock_writer),
        "merged_cluster_writer": ResourceDefinition.hardcoded_resource(mock_writer),
    }
    
    with dg.build_asset_context(resources=resource_defs) as context:
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
        df = pl.DataFrame({
            "store_id": [f"store_{i}" for i in range(10)],
            "feature_1": [i * 10 for i in range(10)],
            "feature_2": [i * i for i in range(10)],
            "feature_3": [None if i % 3 == 0 else i * 5 for i in range(10)],
            "feature_4": ["A" if i % 2 == 0 else "B" for i in range(10)],
        })
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
        assignments = pl.DataFrame({
            "store_id": [f"store_{i}" for i in range(10)],
            "cluster": [i % 3 for i in range(10)],
            "distance": [0.1 * i for i in range(10)],
        })
        sample_assignments[category] = assignments
    
    return sample_assignments 