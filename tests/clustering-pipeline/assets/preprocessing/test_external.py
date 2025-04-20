"""Tests for external preprocessing assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from clustering.pipeline.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)


@pytest.fixture
def mock_external_features() -> pl.DataFrame:
    """Create mock external features data for testing."""
    return pl.DataFrame({
        "store_id": ["store_1", "store_2", "store_3", "store_4"],
        "feature_1": [10.5, 15.2, 8.7, 12.3],
        "feature_2": [1200, 980, 1450, 1100],
        "feature_3": ["A", "B", "A", "C"],
        "latitude": [37.7749, 34.0522, 40.7128, 41.8781],
        "longitude": [-122.4194, -118.2437, -74.0060, -87.6298],
    })


class TestExternalFeaturesData:
    """Tests for external_features_data asset."""
    
    def test_load_external_data(self, mock_execution_context):
        """Test loading external features data from resource."""
        # Configure the mock reader to return test data
        mock_data = pl.DataFrame({
            "store_id": ["store_1", "store_2", "store_3"],
            "feature_1": [10.5, 15.2, 8.7],
            "feature_2": [1200, 980, 1450],
        })
        mock_execution_context.resources.external_features_reader = MagicMock()
        mock_execution_context.resources.external_features_reader.read.return_value = mock_data
        
        # Execute the asset
        result = external_features_data(mock_execution_context)
        
        # Verify result is the data from the reader
        assert isinstance(result, pl.DataFrame)
        assert result.equals(mock_data)
        
        # Verify reader was called
        mock_execution_context.resources.external_features_reader.read.assert_called_once()


class TestPreprocessedExternalData:
    """Tests for preprocessed_external_data asset."""
    
    def test_preprocess_with_defaults(self, mock_execution_context, mock_external_features):
        """Test preprocessing external data with default settings."""
        # Configure default settings in context
        mock_execution_context.resources.config.remove_duplicates = True
        mock_execution_context.resources.config.remove_null_threshold = 0.5
        mock_execution_context.resources.config.standardize_features = True
        
        # Execute the asset
        result = preprocessed_external_data(mock_execution_context, mock_external_features)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        
        # Verify no duplicate stores
        assert result["store_id"].is_unique().all()
        
        # Verify standardized features (should have mean near 0 and std near 1)
        for feature in ["feature_1", "feature_2"]:
            mean = result[feature].mean()
            std = result[feature].std()
            assert abs(mean) < 0.001  # Mean should be close to 0
            assert abs(std - 1.0) < 0.001  # Std should be close to 1
    
    def test_with_duplicate_stores(self, mock_execution_context):
        """Test preprocessing with duplicate store records."""
        # Create data with duplicate stores
        duplicate_data = pl.DataFrame({
            "store_id": ["store_1", "store_1", "store_2", "store_3"],
            "feature_1": [10.5, 11.2, 15.2, 8.7],
            "feature_2": [1200, 1250, 980, 1450],
        })
        
        # Configure settings to remove duplicates
        mock_execution_context.resources.config.remove_duplicates = True
        mock_execution_context.resources.config.standardize_features = False
        
        # Execute the asset
        result = preprocessed_external_data(mock_execution_context, duplicate_data)
        
        # Verify duplicate stores are removed
        assert result.shape[0] == 3  # Should have 3 unique stores
        assert result["store_id"].is_unique().all()
    
    def test_with_high_null_columns(self, mock_execution_context):
        """Test preprocessing with columns having many null values."""
        # Create data with null values
        null_data = pl.DataFrame({
            "store_id": ["store_1", "store_2", "store_3", "store_4"],
            "feature_1": [10.5, 15.2, 8.7, 12.3],
            "feature_2": [1200, 980, 1450, 1100],
            "feature_3": [None, None, None, 5.0],  # 75% null
            "feature_4": [1.0, None, 3.0, 4.0],    # 25% null
        })
        
        # Configure settings with null threshold
        mock_execution_context.resources.config.remove_duplicates = False
        mock_execution_context.resources.config.remove_null_threshold = 0.5
        mock_execution_context.resources.config.standardize_features = False
        
        # Execute the asset
        result = preprocessed_external_data(mock_execution_context, null_data)
        
        # Verify high-null columns are removed but others remain
        assert "feature_3" not in result.columns  # Should be removed (75% null)
        assert "feature_4" in result.columns      # Should remain (25% null)
    
    def test_without_standardization(self, mock_execution_context, mock_external_features):
        """Test preprocessing without standardizing features."""
        # Configure settings without standardization
        mock_execution_context.resources.config.remove_duplicates = False
        mock_execution_context.resources.config.remove_null_threshold = 0.5
        mock_execution_context.resources.config.standardize_features = False
        
        # Get original feature values
        original_feature1 = mock_external_features["feature_1"].to_list()
        
        # Execute the asset
        result = preprocessed_external_data(mock_execution_context, mock_external_features)
        
        # Verify feature values are unchanged
        assert result["feature_1"].to_list() == original_feature1 