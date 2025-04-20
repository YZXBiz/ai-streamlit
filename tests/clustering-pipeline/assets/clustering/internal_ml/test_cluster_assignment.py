"""Tests for internal cluster assignment assets."""

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from clustering.pipeline.assets.clustering.internal_ml.model_training import (
    internal_assign_clusters,
    internal_save_cluster_assignments,
)


@pytest.fixture
def sample_category_data():
    """Sample data for different categories (used as dimensionality reduced features here)."""
    return {
        "category_a": pl.DataFrame(
            {
                "store_id": ["store_0", "store_1", "store_2"],
                "feature_1": [0.5, 0.7, 0.9],
                "feature_2": [0.3, 0.6, 0.2],
            }
        ),
        "category_b": pl.DataFrame(
            {
                "store_id": ["store_3", "store_4", "store_5"],
                "feature_1": [0.4, 0.8, 0.6],
                "feature_2": [0.2, 0.5, 0.7],
            }
        ),
    }


# New fixture for the missing input
@pytest.fixture
def sample_fe_raw_data():
    """Sample raw data matching the structure needed for internal_fe_raw_data."""
    return {
        "category_a": pl.DataFrame(
            {
                "store_id": ["store_0", "store_1", "store_2"],
                "original_feature_1": [5, 7, 9],
                "original_feature_2": [3, 6, 2],
                "some_other_column": ["x", "y", "z"],
            }
        ),
        "category_b": pl.DataFrame(
            {
                "store_id": ["store_3", "store_4", "store_5"],
                "original_feature_1": [4, 8, 6],
                "original_feature_2": [2, 5, 7],
                "some_other_column": ["a", "b", "c"],
            }
        ),
    }


@pytest.fixture
def sample_clustering_results():
    """Sample results from clustering model training."""
    return {
        "category_a": {
            "optimal_clusters": 2,
            "model": "mocked_model_object_a",
            "metrics": {"silhouette": 0.8},
            "experiment_path": "/path/to/exp_a",  # Mock path needed by asset
        },
        "category_b": {
            "optimal_clusters": 3,
            "model": "mocked_model_object_b",
            "metrics": {"silhouette": 0.7},
            "experiment_path": "/path/to/exp_b",  # Mock path needed by asset
        },
    }


@pytest.fixture
def sample_cluster_assignments():
    """Sample cluster assignment results (as expected output from internal_assign_clusters)."""
    return {
        "category_a": pl.DataFrame(
            {
                "store_id": ["store_0", "store_1", "store_2"],
                "original_feature_1": [5, 7, 9],
                "original_feature_2": [3, 6, 2],
                "some_other_column": ["x", "y", "z"],
                "Cluster": [0, 1, 0],  # Added cluster column
            }
        ),
        "category_b": pl.DataFrame(
            {
                "store_id": ["store_3", "store_4", "store_5"],
                "original_feature_1": [4, 8, 6],
                "original_feature_2": [2, 5, 7],
                "some_other_column": ["a", "b", "c"],
                "Cluster": [2, 1, 0],  # Added cluster column
            }
        ),
    }


class TestInternalAssignClusters:
    """Tests for internal_assign_clusters asset."""

    # Patch load_experiment as well, since the asset uses it
    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_assign_clusters(
        self,
        mock_exp_class,
        mock_load_exp,
        mock_execution_context,
        sample_category_data,
        sample_clustering_results,
        sample_fe_raw_data,
    ):
        """Test cluster assignment to data points."""
        mock_exp = MagicMock()
        # Mock both the class instantiation and the load_experiment function
        mock_exp_class.return_value = mock_exp
        mock_load_exp.return_value = mock_exp

        def assign_side_effect(model):
            # assign_model in pycaret might not need the data argument explicitly when called on the experiment object
            # We need to return the structure expected *before* joining back to raw data
            if model == "mocked_model_object_a":
                return pd.DataFrame({"Cluster": [0, 1, 0]})
            elif model == "mocked_model_object_b":
                return pd.DataFrame({"Cluster": [2, 1, 0]})
            return pd.DataFrame()

        mock_exp.assign_model.side_effect = assign_side_effect

        # Execute the asset with all required inputs
        result = internal_assign_clusters(
            mock_execution_context,
            sample_category_data,  # Represents internal_dimensionality_reduced_features
            sample_clustering_results,  # Represents internal_train_clustering_models
            sample_fe_raw_data,  # Represents internal_fe_raw_data
        )

        # Verify load_experiment calls
        assert mock_load_exp.call_count == len(sample_clustering_results)

        # Verify assign_model calls
        assert mock_exp.assign_model.call_count == len(sample_clustering_results)

        # Verify the result structure - should match raw data + Cluster column
        assert isinstance(result, dict)
        for category, expected_raw_df in sample_fe_raw_data.items():
            assert category in result
            result_df = result[category]
            assert isinstance(result_df, pl.DataFrame)
            # Check columns: original raw columns + Cluster
            expected_cols = set(expected_raw_df.columns) | {"Cluster"}
            assert set(result_df.columns) == expected_cols
            # Basic check on row count
            assert len(result_df) == len(expected_raw_df)

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_handle_missing_model(
        self,
        mock_exp_class,
        mock_load_exp,
        mock_execution_context,
        sample_category_data,
        sample_fe_raw_data,
    ):
        """Test handling of missing model for a category."""
        mock_exp = MagicMock()
        mock_load_exp.return_value = mock_exp
        mock_exp_class.return_value = mock_exp

        incomplete_results = {
            "category_a": {
                "optimal_clusters": 3,
                "model": "mocked_model_object",
                "metrics": {"silhouette": 0.75},
                "experiment_path": "/path/to/exp_a",
            },
        }

        mock_assignments_a = pd.DataFrame({"Cluster": [0, 1, 2]})
        mock_exp.assign_model.return_value = mock_assignments_a

        # Pass all required inputs
        result = internal_assign_clusters(
            mock_execution_context, sample_category_data, incomplete_results, sample_fe_raw_data
        )

        assert "category_a" in result
        assert "category_b" not in result  # Should not be processed

        assert isinstance(result["category_a"], pl.DataFrame)
        assert "Cluster" in result["category_a"].columns
        assert set(result["category_a"].columns) == set(
            sample_fe_raw_data["category_a"].columns
        ) | {"Cluster"}


class TestInternalSaveClusterAssignments:
    """Tests for internal_save_cluster_assignments asset."""

    def test_save_assignments(self, mock_execution_context, sample_cluster_assignments):
        """Test saving cluster assignments."""
        result = internal_save_cluster_assignments(
            mock_execution_context, sample_cluster_assignments
        )

        # Use the correct resource key from the context
        writer = mock_execution_context.resources.internal_cluster_assignments

        assert writer.write.call_count == 1
        written_call_args = writer.write.call_args
        assert "data" in written_call_args.kwargs

        # The asset now combines data before writing
        written_data = written_call_args.kwargs["data"]
        assert isinstance(written_data, pl.DataFrame)
        assert "category" in written_data.columns  # Check for the added category column
        # Further checks could verify the combined structure if needed

        # Verify the asset returns None
        assert result is None

    def test_empty_assignments(self, mock_execution_context):
        """Test saving empty cluster assignments."""
        empty_assignments = {}

        result = internal_save_cluster_assignments(mock_execution_context, empty_assignments)

        assert result is None

        # Use the correct resource key from the context
        writer = mock_execution_context.resources.internal_cluster_assignments
        assert writer.write.call_count == 0
