"""Tests for external feature engineering assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dagster import build_op_context

from clustering.pipeline.assets.clustering.external_ml.feature_engineering import (
    external_dimensionality_reduced_features,
    external_fe_raw_data,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_outlier_removed_features,
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
def sample_external_data():
    """Create sample external data for testing."""
    # Create sample data with some missing values and typical features
    df = pl.DataFrame(
        {
            "STORE_NBR": [f"store_{i}" for i in range(10)],
            "feature_1": [i * 10 for i in range(10)],
            "feature_2": [i * i for i in range(10)],
            "feature_3": [None if i % 3 == 0 else i * 5 for i in range(10)],
            "feature_4": ["A" if i % 2 == 0 else "B" for i in range(10)],
        }
    )
    return df


class TestExternalFeRawData:
    """Tests for external_fe_raw_data asset."""

    def test_returns_data_from_reader(self, sample_external_data):
        """Test that it returns data from the reader."""
        # Create a mock reader
        mock_reader = MagicMock()
        mock_reader.read.return_value = sample_external_data

        # Create context with resources
        context = build_op_context(resources={"external_data_reader": mock_reader})

        # Execute the asset
        result = external_fe_raw_data(context)

        # Verify the result is our sample data
        assert isinstance(result, pl.DataFrame)
        assert result.equals(sample_external_data)
        assert "STORE_NBR" in result.columns
        assert "feature_1" in result.columns


class TestExternalFilteredFeatures:
    """Tests for external_filtered_features asset."""

    def test_no_features_to_ignore(self, mock_execution_context, sample_external_data):
        """Test when no features are specified to ignore."""
        # Make sure there are no features to ignore in config
        mock_execution_context.resources.config.ignore_features = []

        # Execute the asset
        result = external_filtered_features(mock_execution_context, sample_external_data)

        # Should return the original data unchanged
        assert result.equals(sample_external_data)

    @patch(
        "clustering.pipeline.assets.clustering.external_ml.feature_engineering.ClusteringExperiment"
    )
    def test_ignore_features(self, mock_exp_class, mock_execution_context, sample_external_data):
        """Test when features are specified to ignore."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create filtered dataframe (without feature_1)
        filtered_df = sample_external_data.drop("feature_1")
        mock_exp.X_train_transformed = filtered_df.to_pandas()

        # Configure ignore_features in the context
        mock_execution_context.resources.config.ignore_features = ["feature_1"]

        # Execute the asset
        result = external_filtered_features(mock_execution_context, sample_external_data)

        # Verify setup was called with correct ignore_features
        mock_exp.setup.assert_called_once()
        args, kwargs = mock_exp.setup.call_args
        assert "ignore_features" in kwargs
        assert kwargs["ignore_features"] == ["feature_1"]

        # Verify the feature was removed
        assert "feature_1" not in result.columns


class TestExternalImputedFeatures:
    """Tests for external_imputed_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.external_ml.feature_engineering.ClusteringExperiment"
    )
    def test_imputation(self, mock_exp_class, mock_execution_context, sample_external_data):
        """Test feature imputation."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a sample imputed dataframe with no nulls
        imputed_df = sample_external_data.clone()
        # Replace nulls with imputed values for feature_3
        imputed_df = imputed_df.with_columns(pl.col("feature_3").fill_null(strategy="mean"))
        mock_exp.X_train_transformed = imputed_df.to_pandas()

        # Configure imputation settings in the context
        mock_execution_context.resources.config.imputation_type = "simple"
        mock_execution_context.resources.config.numeric_imputation = "mean"
        mock_execution_context.resources.config.categorical_imputation = "mode"

        # Execute the asset
        result = external_imputed_features(mock_execution_context, sample_external_data)

        # Verify setup was called with correct imputation parameters
        mock_exp.setup.assert_called_once()
        args, kwargs = mock_exp.setup.call_args
        assert kwargs["imputation_type"] == "simple"
        assert kwargs["numeric_imputation"] == "mean"
        assert kwargs["categorical_imputation"] == "mode"

        # Verify result has no nulls in feature_3
        assert not result.get_column("feature_3").is_null().any()


class TestExternalNormalizedData:
    """Tests for external_normalized_data asset."""

    @patch(
        "clustering.pipeline.assets.clustering.external_ml.feature_engineering.ClusteringExperiment"
    )
    def test_normalization_enabled(
        self, mock_exp_class, mock_execution_context, sample_external_data
    ):
        """Test normalization when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a sample normalized dataframe
        normalized_df = sample_external_data.clone()
        mock_exp.X_train_transformed = normalized_df.to_pandas()

        # Configure normalization settings in the context
        mock_execution_context.resources.config.normalize = True
        mock_execution_context.resources.config.norm_method = "robust"

        # Execute the asset
        result = external_normalized_data(mock_execution_context, sample_external_data)

        # Verify setup was called with correct normalization parameters
        mock_exp.setup.assert_called_once()
        args, kwargs = mock_exp.setup.call_args
        assert kwargs["normalize"] is True
        assert kwargs["normalize_method"] == "robust"

    def test_normalization_disabled(self, mock_execution_context, sample_external_data):
        """Test when normalization is disabled."""
        # Configure normalization settings in the context to disable
        mock_execution_context.resources.config.normalize = False

        # Execute the asset
        result = external_normalized_data(mock_execution_context, sample_external_data)

        # Verify result is the same as input when normalization is disabled
        assert result.equals(sample_external_data)


class TestExternalOutlierRemovedFeatures:
    """Tests for external_outlier_removed_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.external_ml.feature_engineering.ClusteringExperiment"
    )
    def test_outlier_detection_enabled(
        self, mock_exp_class, mock_execution_context, sample_external_data
    ):
        """Test outlier detection when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a sample dataframe with outliers removed (fewer rows)
        outlier_removed_df = sample_external_data.slice(0, 8)  # Simulate removed outliers
        mock_exp.X_train_transformed = outlier_removed_df.to_pandas()

        # Configure outlier detection settings in the context
        mock_execution_context.resources.config.outlier_detection = True
        mock_execution_context.resources.config.outlier_threshold = 0.05
        mock_execution_context.resources.config.outlier_method = "iforest"

        # Execute the asset
        result = external_outlier_removed_features(mock_execution_context, sample_external_data)

        # Verify setup was called with correct outlier detection parameters
        mock_exp.setup.assert_called_once()
        args, kwargs = mock_exp.setup.call_args
        assert kwargs["remove_outliers"] is True
        assert kwargs["outliers_threshold"] == 0.05
        assert kwargs["outliers_method"] == "iforest"

        # Verify result has fewer rows after outlier removal
        assert len(result) == 8  # Based on our mock data

    def test_outlier_detection_disabled(self, mock_execution_context, sample_external_data):
        """Test when outlier detection is disabled."""
        # Configure outlier detection settings in the context to disable
        mock_execution_context.resources.config.outlier_detection = False

        # Execute the asset
        result = external_outlier_removed_features(mock_execution_context, sample_external_data)

        # Verify result is the same as input when outlier detection is disabled
        assert result.equals(sample_external_data)


class TestExternalDimensionalityReducedFeatures:
    """Tests for external_dimensionality_reduced_features asset."""

    @patch(
        "clustering.pipeline.assets.clustering.external_ml.feature_engineering.ClusteringExperiment"
    )
    def test_pca_enabled(self, mock_exp_class, mock_execution_context, sample_external_data):
        """Test PCA dimensionality reduction when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a sample dataframe with reduced dimensions (fewer columns)
        reduced_df = sample_external_data.select(["STORE_NBR", "feature_1", "feature_2"])
        mock_exp.X_train_transformed = reduced_df.to_pandas()

        # Configure PCA settings in the context
        mock_execution_context.resources.config.pca_active = True
        mock_execution_context.resources.config.pca_components = 0.8
        mock_execution_context.resources.config.pca_method = "linear"

        # Execute the asset
        result = external_dimensionality_reduced_features(
            mock_execution_context, sample_external_data
        )

        # Verify setup was called with correct PCA parameters
        mock_exp.setup.assert_called_once()
        args, kwargs = mock_exp.setup.call_args
        assert kwargs["pca"] is True
        assert kwargs["pca_components"] == 0.8
        assert kwargs["pca_method"] == "linear"

        # Verify result has fewer columns after PCA
        assert len(result.columns) < len(sample_external_data.columns)

    def test_pca_disabled(self, mock_execution_context, sample_external_data):
        """Test when PCA is disabled."""
        # Configure PCA settings in the context to disable
        mock_execution_context.resources.config.pca_active = False

        # Execute the asset
        result = external_dimensionality_reduced_features(
            mock_execution_context, sample_external_data
        )

        # Verify result is the same as input when PCA is disabled
        assert result.equals(sample_external_data)
