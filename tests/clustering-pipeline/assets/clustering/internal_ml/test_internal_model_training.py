"""Tests for internal clustering model training assets."""

from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import tempfile
import pytest
import numpy as np
import polars as pl
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from dagster import build_asset_context, ResourceDefinition, AssetExecutionContext, build_op_context

from clustering.pipeline.assets.clustering.internal_ml.model_training import (
    internal_optimal_cluster_counts,
    internal_save_clustering_models,
    internal_train_clustering_models,
    internal_assign_clusters,
    internal_save_cluster_assignments,
    internal_calculate_cluster_metrics,
    internal_generate_cluster_visualizations,
    Defaults,
)


@pytest.fixture
def mock_writer_resource():
    """Mock writer resource with call tracking for visualization paths."""
    mock_writer = MagicMock()
    paths = {
        0: Path("/mock/path/to/visualization_1.png"),
        1: Path("/mock/path/to/visualization_2.png"),
        2: Path("/mock/path/to/visualization_3.png"),
        3: Path("/mock/path/to/visualization_4.png"),
    }
    
    # Set up the write method to return sequential paths
    mock_writer.write.side_effect = lambda *args, **kwargs: str(paths[mock_writer.write.call_count - 1])
    
    return mock_writer


@pytest.fixture
def mock_execution_context(mock_writer_resource) -> AssetExecutionContext:
    """Create a mock execution context for testing Dagster assets."""
    # Create a config mock with return_value that can be configured for tests
    config_mock = MagicMock()
    config_mock.return_value = {
        "clustering": {
            "metrics": ["silhouette", "calinski_harabasz"],
            "methods": ["elbow", "silhouette"],
            "models": ["kmeans", "dbscan"],
            "min_clusters": 2,
            "max_clusters": 10,
            "pca_components": 3,
            "clusters_to_keep": 3
        }
    }
    
    return build_asset_context(
        resources={
            "config": ResourceDefinition.hardcoded_resource(config_mock),
            "internal_model_output": mock_writer_resource,
            "internal_cluster_assignments": mock_writer_resource,
            "logger": ResourceDefinition.hardcoded_resource(MagicMock()),
        }
    )


@pytest.fixture
def sample_category_data():
    """Create sample category data for testing."""
    # Convert pandas to polars
    return {
        "category_a": pl.DataFrame({
            "store_id": ["1", "2", "3", "4", "5"],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "feature_3": [2.0, 2.0, 2.0, 2.0, 2.0],
            "category": ["category_a"] * 5
        }),
        "category_b": pl.DataFrame({
            "store_id": ["6", "7", "8", "9", "10"],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "feature_3": [2.0, 2.0, 2.0, 2.0, 2.0],
            "category": ["category_b"] * 5
        })
    }


@pytest.fixture
def sample_features(sample_category_data):
    """Create sample features dataframe as polars DataFrames."""
    return sample_category_data  # Already converted to polars


@pytest.fixture
def sample_clustering_results():
    """Optimal cluster counts for each category."""
    return {
        "category_a": 3,
        "category_b": 4
    }


@pytest.fixture
def sample_trained_models():
    """Sample trained clustering models for testing."""
    return {
        "category_a": {
            "experiment_path": "mock/path/experiments/category_a",
            "features": ["feature_1", "feature_2", "feature_3"],
            "metrics": {"calinski_harabasz": 95.3, "silhouette": 0.68},
            "model": MagicMock(),  # Use a MagicMock for the model object
            "num_clusters": 3,
            "num_samples": 100
        }
    }


@pytest.fixture
def sample_cluster_assignments():
    """Sample cluster assignments."""
    return {
        "category_a": pl.DataFrame({
            "STORE_NBR": ["1", "2", "3", "4", "5"],
            "Cluster": [0, 1, 0, 2, 1]  # Using the same capitalization as the implementation
        }),
        "category_b": pl.DataFrame({
            "STORE_NBR": ["6", "7", "8", "9", "10"],
            "Cluster": [0, 0, 1, 1, 2]  # Using the same capitalization as the implementation
        })
    }


class TestInternalOptimalClusterCounts:
    """Tests for internal_optimal_cluster_counts asset."""

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    def test_internal_optimal_cluster_counts(self, mock_exp_class, mock_execution_context, sample_features):
        """Test determining optimal cluster counts using PyCaret mocks."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Create a side effect function that returns appropriate values for each call
        # This handles multiple categories and multiple cluster counts
        call_counts = {"category_a": 0, "category_b": 0}
        category_responses = {
            "category_a": [
                pd.DataFrame({"silhouette": [0.5], "calinski_harabasz": [100]}, index=[0]),  # k=2
                pd.DataFrame({"silhouette": [0.7], "calinski_harabasz": [150]}, index=[0]),  # k=3
                pd.DataFrame({"silhouette": [0.6], "calinski_harabasz": [120]}, index=[0]),  # k=4
            ],
            "category_b": [
                pd.DataFrame({"silhouette": [0.4], "calinski_harabasz": [80]}, index=[0]),  # k=2
                pd.DataFrame({"silhouette": [0.5], "calinski_harabasz": [110]}, index=[0]),  # k=3
                pd.DataFrame({"silhouette": [0.45], "calinski_harabasz": [90]}, index=[0]),  # k=4
            ]
        }

        def mock_pull_side_effect(*args, **kwargs):
            current_category = mock_exp._current_category_for_test
            idx = call_counts[current_category]
            call_counts[current_category] += 1
            return category_responses[current_category][idx]

        # Set the side effect to our function
        mock_exp.pull.side_effect = mock_pull_side_effect

        # Simulate setting category during setup
        def setup_side_effect(*args, **kwargs):
            data = kwargs.get('data')
            # Extract category from the 'category' column instead of df.name
            category_name = data['category'].iloc[0]
            mock_exp._current_category_for_test = category_name
            return mock_exp
        mock_exp.setup.side_effect = setup_side_effect
        
        # Set up the configuration with actual integer values
        mock_execution_context.resources.config.min_clusters = 2
        mock_execution_context.resources.config.max_clusters = 4
        mock_execution_context.resources.config.metrics = ["silhouette", "calinski_harabasz"]
        mock_execution_context.resources.config.session_id = 42

        # Call the asset function
        result = internal_optimal_cluster_counts(mock_execution_context, sample_features)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "category_a" in result
        assert "category_b" in result
        assert result["category_a"] == 3  # Based on highest silhouette
        assert result["category_b"] == 3  # Based on highest silhouette


class TestInternalTrainClusteringModels:
    """Tests for internal_train_clustering_models asset."""

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.ClusteringExperiment")
    @patch("tempfile.mkdtemp")  # Keep mkdtemp patch
    def test_train_models(self, mock_mkdtemp, mock_exp_class, mock_execution_context, sample_features, sample_clustering_results):
        """Test training clustering models."""
        # Setup the mock experiment
        mock_exp = MagicMock()
        mock_exp_class.return_value = mock_exp

        # Configure create_model to return a mock model
        mock_model = MagicMock()
        mock_exp.create_model.return_value = mock_model

        # Configure the mock to return cluster metrics
        mock_exp.pull.return_value = pd.DataFrame({"Silhouette": [0.75], "Calinski-Harabasz": [120.5]}, index=[0])
        
        # Mock the get_config method to return appropriate polars dataframes
        mock_exp.get_config.side_effect = lambda key: {
            "X_train": sample_features["category_a"].to_pandas(),  # Convert polars to pandas for PyCaret
            "prep_pipe": MagicMock()  # Mock pipeline object
        }.get(key)

        # Configure tempfile.mkdtemp
        mock_mkdtemp.return_value = "/tmp/mock_tempdir"

        # Call the asset function
        result = internal_train_clustering_models(mock_execution_context, sample_features, sample_clustering_results)

        # Verify PyCaret experiment was created
        assert mock_exp_class.call_count == len(sample_features)

        # Verify models were created and metrics retrieved
        assert mock_exp.create_model.call_count == len(sample_features)
        assert mock_exp.pull.call_count == len(sample_features)

        # Verify the result
        assert isinstance(result, dict)
        assert "category_a" in result
        
        # Check structure of the trained model output
        for category, model_output in result.items():
            assert "experiment_path" in model_output
            assert "features" in model_output
            assert "metrics" in model_output
            assert "model" in model_output
            assert "num_clusters" in model_output
            assert "num_samples" in model_output


class TestInternalSaveClusteringModels:
    """Tests for internal_save_clustering_models asset."""

    def test_save_models(self, mock_execution_context, sample_trained_models):
        """Test saving clustering models."""
        # Access the mock writer resource
        mock_writer = mock_execution_context.resources.internal_model_output
        initial_call_count = mock_writer.write.call_count

        # Execute the asset
        result = internal_save_clustering_models(mock_execution_context, sample_trained_models)

        # Verify the writer was called
        assert mock_writer.write.call_count > initial_call_count

        # Verify the result is a string path
        assert isinstance(result, str)


class TestInternalAssignClusters:
    """Tests for internal_assign_clusters asset."""

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("pathlib.Path.exists")
    @patch("pickle.load") 
    @patch("builtins.open", new_callable=mock_open)
    def test_internal_assign_clusters(self, mock_open, mock_pickle_load, mock_path_exists, mock_load_experiment, mock_execution_context, sample_features, sample_trained_models):
        """Test assigning clusters to stores based on trained models."""
        # Configure mocks
        mock_path_exists.return_value = True
        
        # Create a mock experiment with assign_model method
        mock_exp = MagicMock()
        mock_load_experiment.return_value = mock_exp
        
        # Configure the assign_model to return a DataFrame with cluster assignments
        mock_predictions = pd.DataFrame({
            "store_id": ["1", "2", "3", "4", "5"],
            "Cluster": [0, 1, 0, 2, 1]
        })
        mock_exp.assign_model.return_value = mock_predictions
        
        # Configure pickle.load to return our mock model
        mock_model = MagicMock()
        mock_pickle_load.return_value = mock_model
        
        # Provide the required upstream assets - all three parameters
        result = internal_assign_clusters(
            mock_execution_context, 
            internal_dimensionality_reduced_features=sample_features,
            internal_train_clustering_models=sample_trained_models,
            internal_fe_raw_data=sample_features
        )
        
        # Verify the load_experiment was called
        mock_load_experiment.assert_called()
        
        # Verify the result
        assert isinstance(result, dict)
        assert "category_a" in result
        
        # Verify the result contains the expected columns (Polars DataFrame)
        df = result["category_a"]
        assert isinstance(df, pl.DataFrame)
        assert "store_id" in df.columns
        assert "Cluster" in df.columns  # Using the same capitalization as the implementation

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("pathlib.Path.exists")
    def test_handle_missing_model(self, mock_path_exists, mock_load_experiment, mock_execution_context, sample_features, sample_trained_models):
        """Test handling of missing model files."""
        # Configure mocks
        mock_path_exists.return_value = False  # Simulate missing model file
        mock_load_experiment.side_effect = FileNotFoundError("Mock file not found")

        # Call the function
        result = internal_assign_clusters(
            mock_execution_context,
            internal_dimensionality_reduced_features=sample_features,
            internal_train_clustering_models=sample_trained_models,
            internal_fe_raw_data=sample_features
        )

        # Verify empty result due to missing model
        assert isinstance(result, dict)
        assert len(result) == 0

        # Verify attempt to check file existence
        mock_path_exists.assert_called_once()
        mock_load_experiment.assert_not_called()  # Should not try to load if file doesn't exist

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("pathlib.Path.exists")
    @patch("pickle.load")
    def test_invalid_model_format(
        self, mock_pickle_load, mock_path_exists, mock_load_experiment,
        mock_execution_context, sample_features, sample_trained_models
    ):
        """Test handling of invalid model formats."""
        # Configure mocks
        mock_path_exists.return_value = True
        mock_pickle_load.side_effect = pickle.UnpicklingError("Invalid model format")
        
        # Configure load_experiment to return a mock experiment
        mock_exp = MagicMock()
        mock_load_experiment.return_value = mock_exp
        mock_exp.assign_model.side_effect = ValueError("Model format invalid")

        # Call the function
        result = internal_assign_clusters(
            mock_execution_context,
            internal_dimensionality_reduced_features=sample_features,
            internal_train_clustering_models=sample_trained_models,
            internal_fe_raw_data=sample_features
        )

        # Verify empty result due to invalid model
        assert isinstance(result, dict)
        assert len(result) == 0

        # Verify attempt to load model
        mock_path_exists.assert_called_once()
        mock_load_experiment.assert_called_once()
        # We no longer assert pickle_load was called since our implementation changed

    @patch("clustering.pipeline.assets.clustering.internal_ml.model_training.load_experiment")
    @patch("pathlib.Path.exists")
    @patch("pickle.load")
    def test_mismatched_feature_dimensions(
        self, mock_pickle_load, mock_path_exists, mock_load_experiment,
        mock_execution_context, sample_features, sample_trained_models
    ):
        """Test handling of mismatched feature dimensions between training and prediction."""
        # Configure mocks
        mock_path_exists.return_value = True
        
        # Create mock experiment that raises ValueError on dimension mismatch
        mock_exp = MagicMock()
        mock_load_experiment.return_value = mock_exp
        mock_exp.assign_model.side_effect = ValueError("Feature dimensions do not match training data")
        
        # Configure pickle.load to return mock model
        mock_model = MagicMock()
        mock_pickle_load.return_value = mock_model

        # Call the function
        result = internal_assign_clusters(
            mock_execution_context,
            internal_dimensionality_reduced_features=sample_features,
            internal_train_clustering_models=sample_trained_models,
            internal_fe_raw_data=sample_features
        )

        # Verify empty result due to dimension mismatch
        assert isinstance(result, dict)
        assert len(result) == 0

        # Verify attempts to load and use model
        mock_path_exists.assert_called_once()
        mock_load_experiment.assert_called_once()
        # We no longer assert pickle_load was called since our implementation changed
        mock_exp.assign_model.assert_called_once_with(sample_trained_models["category_a"]["model"])


class TestInternalSaveClusterAssignments:
    """Tests for internal_save_cluster_assignments asset."""

    def test_internal_save_cluster_assignments(self, mock_writer, sample_cluster_assignments):
        """Test saving cluster assignments."""
        # Create context with resources
        context = build_op_context(resources={"internal_cluster_assignments": mock_writer})

        # Call the function under test
        result = internal_save_cluster_assignments(context, sample_cluster_assignments)

        # Verify that the writer was called
        assert mock_writer.write.called

        # Verify the result is a string path
        assert isinstance(result, str)
        assert result == "/mock/path/output.parquet"

        # Check that the call to write included our DataFrame
        args, _ = mock_writer.write.call_args
        assert isinstance(args[0], pl.DataFrame)
        assert "Cluster" in args[0].columns  # Check for cluster column

    def test_empty_assignments(self, mock_writer):
        """Test handling empty cluster assignments."""
        # Create context with resources
        context = build_op_context(resources={"internal_cluster_assignments": mock_writer})

        # Call the function with empty assignments
        empty_assignments = {}
        result = internal_save_cluster_assignments(context, empty_assignments)

        # Function should return a placeholder path string for empty assignments
        assert result == "empty_assignments.parquet"

        # The writer shouldn't be called for empty assignments
        assert not mock_writer.write.called


class TestInternalCalculateClusterMetrics:
    """Tests for internal_calculate_cluster_metrics asset."""

    def test_internal_calculate_cluster_metrics(
        self, mock_execution_context, sample_cluster_assignments, sample_features, sample_trained_models
    ):
        """Test calculating cluster metrics."""
        result = internal_calculate_cluster_metrics(
            mock_execution_context,
            internal_train_clustering_models=sample_trained_models,
            internal_assign_clusters=sample_cluster_assignments,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "category_a" in result
        
        # Verify metrics structure
        category_metrics = result["category_a"]
        assert "num_clusters" in category_metrics
        assert "num_samples" in category_metrics
        assert "cluster_distribution" in category_metrics

    def test_empty_clusters(
        self, mock_execution_context, sample_features, sample_trained_models
    ):
        """Test handling of empty cluster assignments."""
        empty_assignments = {}
        
        result = internal_calculate_cluster_metrics(
            mock_execution_context,
            internal_train_clustering_models=sample_trained_models,
            internal_assign_clusters=empty_assignments,
        )

        # Verify empty result
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_single_store_clusters(
        self, mock_execution_context, sample_features, sample_trained_models
    ):
        """Test handling of clusters with single stores."""
        # Create assignments where each store is in its own cluster
        single_store_assignments = {
            "category_a": pl.DataFrame({
                "STORE_NBR": ["001", "002", "003"],
                "Cluster": [0, 1, 2]
            })
        }

        result = internal_calculate_cluster_metrics(
            mock_execution_context,
            internal_train_clustering_models=sample_trained_models,
            internal_assign_clusters=single_store_assignments,
        )

        # Verify metrics for single-store clusters
        assert isinstance(result, dict)
        assert "category_a" in result
        assert "cluster_distribution" in result["category_a"]
        
        # Verify each cluster has exactly one store
        cluster_dist = result["category_a"]["cluster_distribution"]
        assert all(d["count"] == 1 for d in cluster_dist)

    def test_weighted_metrics_calculation(
        self, mock_execution_context, sample_features, sample_trained_models
    ):
        """Test calculation of metrics with weighted samples."""
        # Create assignments with known weights
        weighted_assignments = {
            "category_a": pl.DataFrame({
                "STORE_NBR": ["001", "002", "003", "004"],
                "Cluster": [0, 0, 1, 1]
            })
        }

        result = internal_calculate_cluster_metrics(
            mock_execution_context,
            internal_train_clustering_models=sample_trained_models,
            internal_assign_clusters=weighted_assignments,
        )

        # Verify metrics calculation with weights
        assert isinstance(result, dict)
        assert "category_a" in result
        
        # Verify cluster distribution
        cluster_dist = result["category_a"]["cluster_distribution"]
        cluster_0_count = next(d["count"] for d in cluster_dist if d["Cluster"] == 0)
        cluster_1_count = next(d["count"] for d in cluster_dist if d["Cluster"] == 1)
        assert cluster_0_count == 2  # Two stores in cluster 0
        assert cluster_1_count == 2  # Two stores in cluster 1


class TestInternalGenerateClusterVisualizations:
    """Tests for internal_generate_cluster_visualizations asset."""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    @patch("tempfile.mkdtemp")
    def test_internal_generate_cluster_visualizations(
        self, mock_mkdtemp, mock_figure, mock_savefig,
        mock_execution_context, sample_cluster_assignments, sample_trained_models
    ):
        """Test generating cluster visualizations."""
        # Configure the mock temp directory
        mock_mkdtemp.return_value = "/mock/path/to/viz"

        # Configure figure and savefig mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig

        # Configure visualization settings in the mock context
        mock_execution_context.resources.config.metadata_detail = "full"
        mock_execution_context.resources.config.visualization_types = ["scatter", "distribution"]

        # Call the asset function
        result = internal_generate_cluster_visualizations(
            mock_execution_context,
            internal_train_clustering_models=sample_trained_models,
            internal_assign_clusters=sample_cluster_assignments,
        )

        # Verify the result structure
        assert isinstance(result, dict)
        assert "category_a" in result
        assert isinstance(result["category_a"], list)
        assert len(result["category_a"]) == 3  # Should generate 3 plots per category

        # Verify savefig was called for each visualization
        assert mock_savefig.call_count == 3  # One call per visualization
        
        # Verify the paths in the result
        for path in result["category_a"]:
            assert path.startswith("plots/category_a_")
            assert path.endswith(".png")
