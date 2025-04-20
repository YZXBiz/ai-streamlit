"""Tests for internal feature engineering assets."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import polars as pl
import pandas as pd
from pycaret.clustering import ClusteringExperiment

from clustering.pipeline.assets.clustering.internal_ml.feature_engineering import (
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
    internal_dimensionality_reduced_features,
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
def mock_context_with_data(mock_execution_context, sample_category_data):
    """Create mock context with sample data in the reader."""
    # Update the mock reader to return our sample data
    mock_execution_context.resources.sales_by_category_reader.data = sample_category_data
    return mock_execution_context


class TestInternalFeRawData:
    """Tests for internal_fe_raw_data asset."""
    
    def test_returns_data_from_reader(self, mock_context_with_data, sample_category_data):
        """Test that it returns data from the reader."""
        # Execute the asset
        result = internal_fe_raw_data(mock_context_with_data)
        
        # Verify the result matches our sample data
        assert result == sample_category_data
        assert "category_a" in result
        assert "category_b" in result
        
        # Verify the data structure
        for category, df in result.items():
            assert isinstance(df, pl.DataFrame)
            assert "store_id" in df.columns
            assert "feature_1" in df.columns


class TestInternalFilteredFeatures:
    """Tests for internal_filtered_features asset."""
    
    def test_no_features_to_ignore(self, mock_context_with_data, sample_category_data):
        """Test when no features are specified to ignore."""
        # Execute the asset
        result = internal_filtered_features(mock_context_with_data, sample_category_data)
        
        # Should return the original data unchanged
        assert result == sample_category_data
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_ignore_features(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test when features are specified to ignore."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Configure the transformed data to return
        mock_df = sample_category_data["category_a"].drop("feature_1")
        mock_exp.X_train_transformed = mock_df.to_pandas()
        
        # Configure ignore_features in the context
        mock_context_with_data.resources.config.ignore_features = ["feature_1"]
        
        # Execute the asset
        result = internal_filtered_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with correct ignore_features
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert "ignore_features" in kwargs
            assert "feature_1" in kwargs["ignore_features"]
        
        # Verify the correct columns are in the result
        for category in result:
            if category == "category_a":
                # This is our mocked result which should have feature_1 removed
                assert "feature_1" not in result[category].columns


class TestInternalImputedFeatures:
    """Tests for internal_imputed_features asset."""
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_imputation(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test feature imputation."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample imputed dataframe
        imputed_df = sample_category_data["category_a"].clone()
        # In reality, imputation would fill null values
        mock_exp.X_train_transformed = imputed_df.to_pandas()
        
        # Configure imputation settings in the context
        mock_context_with_data.resources.config.imputation_type = "simple"
        mock_context_with_data.resources.config.numeric_imputation = "mean"
        mock_context_with_data.resources.config.categorical_imputation = "mode"
        
        # Execute the asset
        result = internal_imputed_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with correct imputation parameters
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["imputation_type"] == "simple"
            assert kwargs["numeric_imputation"] == "mean"
            assert kwargs["categorical_imputation"] == "mode"
        
        # Verify result contains imputed data
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)


class TestInternalNormalizedData:
    """Tests for internal_normalized_data asset."""
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_normalization_enabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test normalization when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample normalized dataframe
        normalized_df = sample_category_data["category_a"].clone()
        mock_exp.X_train_transformed = normalized_df.to_pandas()
        
        # Configure normalization settings in the context
        mock_context_with_data.resources.config.normalize = True
        mock_context_with_data.resources.config.norm_method = "robust"
        
        # Execute the asset
        result = internal_normalized_data(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with correct normalization parameters
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["normalize"] == True
            assert kwargs["normalize_method"] == "robust"
        
        # Verify result contains normalized data
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_normalization_disabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test when normalization is disabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample non-normalized dataframe
        non_normalized_df = sample_category_data["category_a"].clone()
        mock_exp.X_train_transformed = non_normalized_df.to_pandas()
        
        # Configure normalization settings in the context
        mock_context_with_data.resources.config.normalize = False
        
        # Execute the asset
        result = internal_normalized_data(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with normalization disabled
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["normalize"] == False
        
        # Verify result contains non-normalized data
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)


class TestInternalOutlierRemovedFeatures:
    """Tests for internal_outlier_removed_features asset."""
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_outlier_detection_enabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test outlier detection when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample dataframe with outliers removed
        outlier_removed_df = sample_category_data["category_a"].clone()
        # In reality, outlier removal would reduce the number of rows
        mock_exp.X_train_transformed = outlier_removed_df.to_pandas()
        
        # Configure outlier detection settings in the context
        mock_context_with_data.resources.config.outlier_detection = True
        mock_context_with_data.resources.config.outlier_threshold = 0.05
        mock_context_with_data.resources.config.outlier_method = "iforest"
        
        # Execute the asset
        result = internal_outlier_removed_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with correct outlier detection parameters
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["remove_outliers"] == True
            assert kwargs["outliers_threshold"] == 0.05
            assert kwargs["outliers_method"] == "iforest"
        
        # Verify result contains data with outliers removed
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_outlier_detection_disabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test when outlier detection is disabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample dataframe without outlier removal
        outlier_df = sample_category_data["category_a"].clone()
        mock_exp.X_train_transformed = outlier_df.to_pandas()
        
        # Configure outlier detection settings in the context
        mock_context_with_data.resources.config.outlier_detection = False
        
        # Execute the asset
        result = internal_outlier_removed_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with outlier detection disabled
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["remove_outliers"] == False
        
        # Verify result contains data without outlier removal
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)


class TestInternalDimensionalityReducedFeatures:
    """Tests for internal_dimensionality_reduced_features asset."""
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_pca_enabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test PCA when enabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample PCA-reduced dataframe
        pca_df = sample_category_data["category_a"].clone()
        # In reality, PCA would reduce the number of features
        mock_exp.X_train_transformed = pca_df.to_pandas()
        
        # Configure PCA settings in the context
        mock_context_with_data.resources.config.pca_active = True
        mock_context_with_data.resources.config.pca_components = 0.8
        mock_context_with_data.resources.config.pca_method = "linear"
        
        # Execute the asset
        result = internal_dimensionality_reduced_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with correct PCA parameters
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["pca"] == True
            assert kwargs["pca_components"] == 0.8
            assert kwargs["pca_method"] == "linear"
        
        # Verify result contains PCA-reduced data
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame)
    
    @patch("clustering.pipeline.assets.clustering.internal_ml.feature_engineering.ClusteringExperiment")
    def test_pca_disabled(self, mock_exp_class, mock_context_with_data, sample_category_data):
        """Test when PCA is disabled."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create a sample non-PCA dataframe
        non_pca_df = sample_category_data["category_a"].clone()
        mock_exp.X_train_transformed = non_pca_df.to_pandas()
        
        # Configure PCA settings in the context
        mock_context_with_data.resources.config.pca_active = False
        
        # Execute the asset
        result = internal_dimensionality_reduced_features(mock_context_with_data, sample_category_data)
        
        # Verify setup was called with PCA disabled
        mock_exp.setup.assert_called()
        setup_calls = mock_exp.setup.call_args_list
        for call in setup_calls:
            args, kwargs = call
            assert kwargs["pca"] == False
        
        # Verify result contains non-PCA data
        assert "category_a" in result
        assert isinstance(result["category_a"], pl.DataFrame) 