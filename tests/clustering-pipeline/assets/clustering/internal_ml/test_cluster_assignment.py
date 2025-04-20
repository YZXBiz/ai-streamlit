"""Tests for internal cluster assignment assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pycaret.clustering import ClusteringExperiment

from clustering.pipeline.assets.clustering import (
    internal_assign_clusters,
    internal_save_cluster_assignments,
)


class TestInternalAssignClusters:
    """Tests for internal_assign_clusters asset."""
    
    @patch("clustering.pipeline.assets.clustering.ClusteringExperiment")
    def test_assign_clusters(self, mock_exp_class, mock_execution_context, sample_category_data, sample_clustering_results):
        """Test cluster assignment to data points."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Configure the mock to return assignments
        mock_assignments = pl.DataFrame({
            "store_id": ["store_0", "store_1", "store_2", "store_3", "store_4"],
            "cluster": [0, 1, 2, 0, 1],
            "distance": [0.1, 0.2, 0.15, 0.18, 0.22],
        })
        # Convert to pandas for the mock return value
        mock_exp.assign_model.return_value = mock_assignments.to_pandas()
        
        # Execute the asset
        result = internal_assign_clusters(
            mock_execution_context,
            sample_category_data,
            sample_clustering_results
        )
        
        # Verify PyCaret experiment was created for each category
        assert mock_exp_class.call_count >= len(sample_category_data)
        
        # Verify assign_model was called with the correct model
        for call in mock_exp.assign_model.call_args_list:
            args, kwargs = call
            # First argument should be the model from sample_clustering_results
            assert args[0] is not None
        
        # Verify the result
        assert isinstance(result, dict)
        for category in sample_category_data:
            assert category in result
            assert isinstance(result[category], pl.DataFrame)
            assert "store_id" in result[category].columns
            assert "cluster" in result[category].columns
    
    @patch("clustering.pipeline.assets.clustering.ClusteringExperiment")
    def test_handle_missing_model(self, mock_exp_class, mock_execution_context, sample_category_data):
        """Test handling of missing model for a category."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp
        
        # Create results with missing model for one category
        incomplete_results = {
            "category_a": {
                "optimal_clusters": 3,
                "model": "mocked_model_object",
                "metrics": {"silhouette": 0.75},
            },
            # Missing category_b
        }
        
        # Configure the mock to return assignments
        mock_assignments = pl.DataFrame({
            "store_id": ["store_0", "store_1", "store_2"],
            "cluster": [0, 1, 2],
            "distance": [0.1, 0.2, 0.15],
        })
        mock_exp.assign_model.return_value = mock_assignments.to_pandas()
        
        # Execute the asset
        result = internal_assign_clusters(
            mock_execution_context,
            sample_category_data,
            incomplete_results
        )
        
        # Verify result only contains category with model
        assert "category_a" in result
        assert "category_b" not in result
        
        # Verify category in result has correct format
        assert isinstance(result["category_a"], pl.DataFrame)
        assert "store_id" in result["category_a"].columns
        assert "cluster" in result["category_a"].columns


class TestInternalSaveClusterAssignments:
    """Tests for internal_save_cluster_assignments asset."""
    
    def test_save_assignments(self, mock_execution_context, sample_cluster_assignments):
        """Test saving cluster assignments."""
        # Execute the asset
        result = internal_save_cluster_assignments(mock_execution_context, sample_cluster_assignments)
        
        # Get the writer that should have been used
        writer = mock_execution_context.resources.cluster_assignments_writer
        
        # Verify write was called
        assert writer.written_count > 0
        
        # Verify data was written
        assert writer.written_data[0] == sample_cluster_assignments
        
        # Verify the result is passed through unchanged
        assert result == sample_cluster_assignments
    
    def test_empty_assignments(self, mock_execution_context):
        """Test saving empty cluster assignments."""
        # Create empty assignments
        empty_assignments = {}
        
        # Execute the asset
        result = internal_save_cluster_assignments(mock_execution_context, empty_assignments)
        
        # Verify result is the empty dictionary
        assert result == empty_assignments
        
        # Writer should not have been called for empty data
        writer = mock_execution_context.resources.cluster_assignments_writer
        assert writer.written_count == 0 