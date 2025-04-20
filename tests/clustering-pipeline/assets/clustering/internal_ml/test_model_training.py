"""Tests for internal clustering model training assets."""

from unittest.mock import MagicMock, patch

from clustering.pipeline.assets.clustering.internal_ml.model_training import (
    internal_optimal_cluster_counts,
    internal_save_clustering_models,
    internal_train_clustering_models,
)


class TestInternalOptimalClusterCounts:
    """Tests for internal_optimal_cluster_counts asset."""

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_optimal_cluster_counts(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test determining optimal cluster counts."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Configure the mock to return specific cluster counts
        mock_exp.get_metrics.return_value = {
            "cluster_range": [2, 3, 4, 5, 6],
            "silhouette": [0.5, 0.7, 0.6, 0.55, 0.5],
            "calinski_harabasz": [100, 150, 130, 120, 110],
        }

        # Configure the context resources
        mock_execution_context.resources.config.cluster_range_start = 2
        mock_execution_context.resources.config.cluster_range_end = 6
        mock_execution_context.resources.config.optimize_metric = "silhouette"

        # Execute the asset
        result = internal_optimal_cluster_counts(mock_execution_context, sample_category_data)

        # Verify PyCaret experiment was created
        mock_exp_class.assert_called()

        # Verify cluster metrics were retrieved
        mock_exp.get_metrics.assert_called()

        # Verify the result
        assert isinstance(result, dict)
        assert "category_a" in result
        assert "category_b" in result

        # Check that each category has an optimal cluster count
        for _category, data in result.items():
            assert "optimal_clusters" in data
            assert "metrics" in data
            assert (
                data["optimal_clusters"] == 3
            )  # Mock data has max silhouette at index 1 (3 clusters)
            assert "silhouette" in data["metrics"]
            assert "calinski_harabasz" in data["metrics"]

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_different_optimization_metric(
        self, mock_exp_class, mock_execution_context, sample_category_data
    ):
        """Test using a different optimization metric."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Configure the mock to return specific cluster counts
        mock_exp.get_metrics.return_value = {
            "cluster_range": [2, 3, 4, 5, 6],
            "silhouette": [0.5, 0.7, 0.6, 0.55, 0.5],
            "calinski_harabasz": [100, 150, 200, 170, 120],  # Max at index 2 (4 clusters)
        }

        # Configure the context resources
        mock_execution_context.resources.config.cluster_range_start = 2
        mock_execution_context.resources.config.cluster_range_end = 6
        mock_execution_context.resources.config.optimize_metric = "calinski_harabasz"

        # Execute the asset
        result = internal_optimal_cluster_counts(mock_execution_context, sample_category_data)

        # Verify the correct optimization metric was used
        for _category, data in result.items():
            assert (
                data["optimal_clusters"] == 4
            )  # Should be index 2 (4 clusters) for calinski_harabasz


class TestInternalTrainClusteringModels:
    """Tests for internal_train_clustering_models asset."""

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_train_models(
        self,
        mock_exp_class,
        mock_execution_context,
        sample_category_data,
        sample_clustering_results,
    ):
        """Test training clustering models."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Configure create_model to return a mock model
        mock_model = MagicMock()
        mock_exp.create_model.return_value = mock_model

        # Configure the mock to return cluster metrics
        mock_exp.pull.return_value = {
            "Silhouette": 0.75,
            "Calinski-Harabasz": 120.5,
        }

        # Execute the asset
        result = internal_train_clustering_models(
            mock_execution_context, sample_category_data, sample_clustering_results
        )

        # Verify PyCaret experiment was created for each category
        assert mock_exp_class.call_count >= len(sample_category_data)

        # Verify create_model was called with correct parameters
        for call in mock_exp.create_model.call_args_list:
            args, kwargs = call
            # First argument should be algorithm name
            assert args[0] in ["kmeans", "kmedoids", "ap"]  # Common algorithms
            # n_clusters should be passed
            assert "n_clusters" in kwargs

        # Verify the result
        assert isinstance(result, dict)
        for category in sample_category_data:
            assert category in result
            assert "model" in result[category]
            assert "metrics" in result[category]
            assert "optimal_clusters" in result[category]


class TestInternalSaveClusteringModels:
    """Tests for internal_save_clustering_models asset."""

    def test_save_models(self, mock_execution_context, sample_clustering_results):
        """Test saving clustering models."""
        # Execute the asset
        result = internal_save_clustering_models(mock_execution_context, sample_clustering_results)

        # Get the writer that should have been used
        writer = mock_execution_context.resources.clustering_models_writer

        # Verify write was called
        assert writer.written_count > 0

        # Verify data was written
        assert writer.written_data[0] == sample_clustering_results

        # Verify the result is passed through unchanged
        assert result == sample_clustering_results
