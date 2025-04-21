"""Tests for internal feature engineering assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dagster import build_asset_context, ResourceDefinition

from clustering.pipeline.assets.clustering.internal_ml.feature_engineering import (
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
    internal_dimensionality_reduced_features,
    internal_feature_metadata,
)


@pytest.fixture
def mock_pycaret_setup():
    """Mock PyCaret setup for testing."""
    with patch("pycaret.clustering.ClusteringExperiment") as mock_exp:
        # Configure the mock experiment
        mock_instance = MagicMock()
        mock_instance.X_train_transformed = MagicMock()

        # Make the constructor return our controlled instance
        mock_exp.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_execution_context():
    """Create a mock execution context for testing Dagster assets."""

    class MockReader:
        def read(self):
            return {
                "category_a": pl.DataFrame(
                    {
                        "STORE_NBR": ["1", "2", "3"],
                        "WEEKLY_SALES": [1000, 2000, 3000],
                        "IGNORED_FEATURE": [1, 2, 3],
                    }
                )
            }

    class MockConfig:
        ignore_features = ["IGNORED_FEATURE"]
        imputation_type = "simple"
        numeric_imputation = "mean"
        categorical_imputation = "mode"
        normalize = True
        norm_method = "robust"
        outlier_detection = True
        outlier_threshold = 0.05
        outlier_method = "iforest"
        pca_active = True
        pca_components = 0.8
        pca_method = "linear"
        metadata_detail = "full"

    class MockLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    # Define the resources that will be available in the context
    resource_defs = {
        "config": ResourceDefinition.hardcoded_resource(MockConfig()),
        "sales_by_category_reader": ResourceDefinition.hardcoded_resource(MockReader()),
        "logger": ResourceDefinition.hardcoded_resource(MockLogger()),
    }

    # Create a proper Dagster asset execution context with our mock resources
    with build_asset_context(resources=resource_defs) as context:
        yield context


@pytest.fixture
def mock_context_with_data(mock_execution_context, sample_category_data):
    """Create mock context with sample data in the reader."""
    # This is no longer a simple attribute assignment, we need to adjust the approach
    # to work with the proper Dagster execution context

    # Method 1: Patch the read method in this fixture to return our sample data
    original_read = mock_execution_context.resources.sales_by_category_reader.read

    def patched_read(*args, **kwargs):
        return sample_category_data

    mock_execution_context.resources.sales_by_category_reader.read = patched_read

    yield mock_execution_context

    # Restore original method after test
    mock_execution_context.resources.sales_by_category_reader.read = original_read


@pytest.fixture
def sample_data():
    """Create sample pandas DataFrame for compatibility with PyCaret."""
    return pd.DataFrame(
        {
            "STORE_NBR": ["1", "2", "3"],
            "WEEKLY_SALES": [1000, 2000, 3000],
            "STORE_SIZE": [100, 200, 300],
            "AVG_TEMPERATURE": [70, 75, 80],
            "FUEL_PRICE": [2.5, 2.7, 2.9],
            "UNEMPLOYMENT": [5.0, 5.5, 6.0],
            "URBAN_RURAL": ["URBAN", "RURAL", "URBAN"],
        }
    )


@pytest.fixture
def sample_category_data():
    """Create sample category data using polars DataFrames."""
    return {
        "category_a": pl.DataFrame(
            {
                "STORE_NBR": ["1", "2", "3"],
                "WEEKLY_SALES": [1000, 2000, 3000],
                "IGNORED_FEATURE": [1, 2, 3],
            }
        )
    }


class TestInternalFeRawData:
    """Tests for internal_fe_raw_data asset."""

    def test_internal_fe_raw_data(self, mock_execution_context, sample_category_data):
        """Test that the internal_fe_raw_data asset returns data from the reader."""
        # Replace reader's read method with a mock that returns sample data
        original_read = mock_execution_context.resources.sales_by_category_reader.read
        mock_execution_context.resources.sales_by_category_reader.read = (
            lambda: sample_category_data
        )

        # Call the asset function
        result = internal_fe_raw_data(mock_execution_context)

        # Verify the result
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert result == sample_category_data  # Should return original data

        # Restore original method
        mock_execution_context.resources.sales_by_category_reader.read = original_read


class TestInternalFilteredFeatures:
    """Tests for internal_filtered_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_filtered_features(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test filtering features based on configuration."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a dictionary of polars DataFrames for the input
        input_dict = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["1", "2", "3"],
                    "WEEKLY_SALES": [1000, 2000, 3000],
                    "IGNORED_FEATURE": [1, 2, 3],
                }
            )
        }

        # Configure mock for the expected filtered output
        filtered_df = pd.DataFrame(
            {
                "STORE_NBR": ["1", "2", "3"],
                "WEEKLY_SALES": [1000, 2000, 3000],
            }
        )

        # Mock conversion inside the asset
        mock_exp.setup.return_value = mock_exp
        mock_exp.X = filtered_df
        # Important: mock X_train_transformed as a pandas DataFrame
        mock_exp.X_train_transformed = filtered_df

        # Call the asset function with polars DataFrame input
        result = internal_filtered_features(mock_execution_context, input_dict)

        # Verify the result structure - should return Polars DataFrames
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert "IGNORED_FEATURE" not in result["category_a"].columns
        assert "STORE_NBR" in result["category_a"].columns
        assert "WEEKLY_SALES" in result["category_a"].columns


class TestInternalImputedFeatures:
    """Tests for internal_imputed_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_imputed_features(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test imputing missing values in features."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a dictionary of polars DataFrames for the input
        input_dict = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["1", "2", "3"],
                    "WEEKLY_SALES": [1000, 2000, 3000],
                }
            )
        }

        # Configure mock for the expected imputed output
        mock_exp.setup.return_value = mock_exp
        mock_exp.X_train_transformed = input_dict["category_a"].to_pandas()

        # Call the asset function with polars DataFrame input
        result = internal_imputed_features(mock_execution_context, input_dict)

        # Verify the result structure - should return Polars DataFrames
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert "STORE_NBR" in result["category_a"].columns
        assert "WEEKLY_SALES" in result["category_a"].columns


class TestInternalNormalizedData:
    """Tests for internal_normalized_data asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_normalized_data(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test normalizing feature values."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a dictionary of polars DataFrames for the input
        input_dict = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["1", "2", "3"],
                    "WEEKLY_SALES": [1000, 2000, 3000],
                }
            )
        }

        # Configure mock for the expected normalized output
        mock_exp.setup.return_value = mock_exp
        normalized_df = pd.DataFrame(
            {
                "STORE_NBR": ["1", "2", "3"],
                "WEEKLY_SALES": [0.0, 0.5, 1.0],  # Normalized values
            }
        )
        mock_exp.X_train_transformed = normalized_df

        # Call the asset function with polars DataFrame input
        result = internal_normalized_data(mock_execution_context, input_dict)

        # Verify the result structure - should return Polars DataFrames
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert "STORE_NBR" in result["category_a"].columns
        assert "WEEKLY_SALES" in result["category_a"].columns


class TestInternalOutlierRemovedFeatures:
    """Tests for internal_outlier_removed_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_outlier_removed_features(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test removing outliers from features."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a dictionary of polars DataFrames for the input
        input_dict = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["1", "2", "3"],
                    "WEEKLY_SALES": [1000, 2000, 3000],
                }
            )
        }

        # Configure mock for expected output after outlier removal
        mock_exp.setup.return_value = mock_exp
        # Simulate data with outliers removed (e.g., fewer rows)
        outlier_removed_df = pd.DataFrame(
            {
                "STORE_NBR": ["1", "2"],
                "WEEKLY_SALES": [1000, 2000],
            }
        )
        mock_exp.X_train_transformed = outlier_removed_df

        # Call the asset function with polars DataFrame input
        result = internal_outlier_removed_features(mock_execution_context, input_dict)

        # Verify the result structure - should return Polars DataFrames
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert "STORE_NBR" in result["category_a"].columns
        assert "WEEKLY_SALES" in result["category_a"].columns


class TestInternalDimensionalityReducedFeatures:
    """Tests for internal_dimensionality_reduced_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_dimensionality_reduced_features(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test reducing dimensionality of features."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a dictionary of polars DataFrames for the input
        input_dict = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["1", "2", "3"],
                    "WEEKLY_SALES": [1000, 2000, 3000],
                }
            )
        }

        # Configure mock for expected output after PCA
        mock_exp.setup.return_value = mock_exp
        # Simulate dimensionality reduced data (e.g., fewer columns)
        pca_df = pd.DataFrame(
            {
                "STORE_NBR": ["1", "2", "3"],
                "PC1": [0.1, 0.2, 0.3],  # PCA components
            }
        )
        mock_exp.X_train_transformed = pca_df

        # Call the asset function with polars DataFrame input
        result = internal_dimensionality_reduced_features(mock_execution_context, input_dict)

        # Verify the result structure - should return Polars DataFrames
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
        assert len(result["category_a"].columns) > 0


class TestInternalFeatureMetadata:
    """Tests for internal_feature_metadata asset."""

    @patch(
        "clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment"
    )
    def test_internal_feature_metadata(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test generating feature metadata."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create polars DataFrames for the input
        input_dict = sample_category_data

        # Configure feature metadata and other attributes PyCaret might provide
        mock_exp.setup.return_value = mock_exp
        mock_exp.get_params.return_value = {"normalize": True, "pca": 0.8}

        # Mock the data attributes that would be available
        mock_exp.data = input_dict["category_a"].to_pandas()

        # Configure metadata detail level
        mock_execution_context.resources.config.feature_metadata_detail = "full"
        mock_execution_context.resources.config.pca_active = True
        mock_execution_context.resources.config.pca_components = 0.8
        mock_execution_context.resources.config.normalize = True
        mock_execution_context.resources.config.norm_method = "robust"
        mock_execution_context.resources.config.outlier_detection = True
        mock_execution_context.resources.config.outlier_threshold = 0.05
        mock_execution_context.resources.config.algorithm = "kmeans"

        # Call the asset function
        result = internal_feature_metadata(mock_execution_context, input_dict)

        # Verify the result
        assert isinstance(result, dict)
        assert "category_a" in result

        # Verify the metadata structure
        metadata = result["category_a"]
        assert "config" in metadata
        assert "null_counts" in metadata
        assert "correlations" in metadata
        assert "sample" in metadata

        # Verify specific config settings are captured
        assert metadata["config"]["pca"]["enabled"] is True
        assert metadata["config"]["pca"]["variance"] == 0.8
        assert metadata["config"]["normalization"]["enabled"] is True
        assert metadata["config"]["normalization"]["method"] == "robust"
