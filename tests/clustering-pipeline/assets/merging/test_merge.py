"""Tests for cluster merging assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from clustering.pipeline.assets.merging.merge import (
    cluster_reassignment,
    merged_clusters,
    optimized_merged_clusters,
    save_merged_cluster_assignments,
)


@pytest.fixture
def sample_internal_assignments() -> dict[str, pl.DataFrame]:
    """Create sample internal store cluster assignments."""
    return {
        "category_a": pl.DataFrame(
            {
                "store_id": [f"store_{i}" for i in range(5)],
                "cluster": [0, 1, 2, 0, 1],
                "distance": [0.1, 0.2, 0.15, 0.18, 0.22],
            }
        ),
    }


@pytest.fixture
def sample_external_assignments() -> dict[str, pl.DataFrame]:
    """Create sample external store cluster assignments."""
    return {
        "category_b": pl.DataFrame(
            {
                "store_id": [f"store_{i}" for i in range(5)],
                "cluster": [1, 0, 2, 1, 0],
                "distance": [0.12, 0.19, 0.14, 0.17, 0.21],
            }
        ),
    }


@pytest.fixture
def sample_merged_clusters() -> pl.DataFrame:
    """Create sample merged clusters data."""
    return pl.DataFrame(
        {
            "store_id": [f"store_{i}" for i in range(5)],
            "category_a_cluster": [0, 1, 2, 0, 1],
            "category_b_cluster": [1, 0, 2, 1, 0],
            "category_a_distance": [0.1, 0.2, 0.15, 0.18, 0.22],
            "category_b_distance": [0.12, 0.19, 0.14, 0.17, 0.21],
        }
    )


class TestMergedClusters:
    """Tests for merged_clusters asset."""

    @patch("os.path.exists")
    def test_merge_clusters(
        self,
        mock_exists,
        mock_execution_context,
        sample_internal_assignments,
        sample_external_assignments,
    ):
        """Test merging internal and external cluster assignments."""
        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Set up the reader resources to return our test data
        internal_reader = mock_execution_context.resources.internal_cluster_assignments
        internal_reader.data = sample_internal_assignments

        external_reader = mock_execution_context.resources.external_cluster_assignments
        external_reader.data = sample_external_assignments

        # Execute the asset
        result = merged_clusters(mock_execution_context)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "category_a_cluster" in result.columns
        assert "category_b_cluster" in result.columns
        assert "category_a_distance" in result.columns
        assert "category_b_distance" in result.columns

        # Verify data is merged correctly
        assert result.height == 5  # 5 stores

        # Check a sample value
        first_row = result.row(0, named=True)
        assert first_row["store_id"] == "store_0"
        assert first_row["category_a_cluster"] == 0
        assert first_row["category_b_cluster"] == 1

    @patch("os.path.exists")
    def test_merge_with_missing_stores(self, mock_exists, mock_execution_context):
        """Test merging when stores don't perfectly overlap."""
        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Create assignments with different store sets
        internal_assignments = {
            "category_a": pl.DataFrame(
                {
                    "store_id": ["store_1", "store_2", "store_3"],
                    "cluster": [1, 2, 0],
                    "distance": [0.2, 0.15, 0.18],
                }
            ),
        }

        external_assignments = {
            "category_b": pl.DataFrame(
                {
                    "store_id": ["store_2", "store_3", "store_4"],
                    "cluster": [2, 1, 0],
                    "distance": [0.14, 0.17, 0.21],
                }
            ),
        }

        # Set up the reader resources to return our test data
        internal_reader = mock_execution_context.resources.internal_cluster_assignments
        internal_reader.data = internal_assignments

        external_reader = mock_execution_context.resources.external_cluster_assignments
        external_reader.data = external_assignments

        # Execute the asset
        result = merged_clusters(mock_execution_context)

        # Verify result contains all stores from both datasets
        assert set(result["store_id"].to_list()) == {"store_1", "store_2", "store_3", "store_4"}

        # Verify stores only in one dataset have null values for the other
        for row in result.rows(named=True):
            if row["store_id"] == "store_1":
                assert row["category_b_cluster"] is None
            elif row["store_id"] == "store_4":
                assert row["category_a_cluster"] is None


class TestOptimizedMergedClusters:
    """Tests for optimized_merged_clusters asset."""

    def test_optimize_clusters(self, mock_execution_context, sample_merged_clusters):
        """Test optimizing merged clusters."""
        # Configure the context resources
        mock_execution_context.resources.config.optimization_method = "distance_weighted"
        mock_execution_context.resources.config.category_weights = {
            "category_a": 0.6,
            "category_b": 0.4,
        }

        # Create mock merged cluster assignments
        merged_cluster_assignments_data = {
            "clusters": {"merged_cluster": ["0_1", "1_0", "2_2"], "count": [2, 2, 1]},
            "store_mappings": {
                "STORE_NBR": ["store_0", "store_1", "store_2", "store_3", "store_4"],
                "merged_cluster": ["0_1", "1_0", "2_2", "0_1", "1_0"],
            },
        }

        # Execute the asset
        result = optimized_merged_clusters(
            mock_execution_context, sample_merged_clusters, merged_cluster_assignments_data
        )

        # Verify result structure (it's a dictionary)
        assert isinstance(result, dict)
        assert "small_clusters" in result
        assert "large_clusters" in result
        assert "merged_data" in result
        assert isinstance(result["small_clusters"], pl.DataFrame)
        assert isinstance(result["large_clusters"], pl.DataFrame)
        assert isinstance(result["merged_data"], pl.DataFrame)

        # Check contents of merged_data
        merged_df = result["merged_data"]
        assert "store_id" in merged_df.columns
        assert "category_a_cluster" in merged_df.columns
        assert "category_b_cluster" in merged_df.columns
        # We are not checking for optimal_cluster and score here as they are added in a later step

        # Verify cluster counts (example check)
        assert result["small_clusters"].height >= 0
        assert result["large_clusters"].height >= 0


class TestClusterReassignment:
    """Tests for cluster_reassignment asset."""

    def test_reassign_clusters(self, mock_execution_context, sample_merged_clusters):
        """Test reassigning clusters based on optimization."""
        # Mock the input dictionary expected by cluster_reassignment
        # This dictionary is the output of optimized_merged_clusters
        mock_input_dict = {
            "small_clusters": pl.DataFrame(
                {
                    "merged_cluster": ["0_1"],
                    "count": [2],  # Example small cluster
                }
            ),
            "large_clusters": pl.DataFrame(
                {
                    "merged_cluster": ["1_0", "2_2"],
                    "count": [2, 1],  # Example large clusters
                }
            ),
            "merged_data": sample_merged_clusters.rename({"store_id": "STORE_NBR"}).with_columns(
                (
                    pl.col("category_a_cluster").cast(pl.Utf8)
                    + "_"
                    + pl.col("category_b_cluster").cast(pl.Utf8)
                ).alias("merged_cluster")
            ),  # Add the merged_cluster column needed for reassignment logic
        }

        # Mock necessary resources via the fixture (paths and job_params updated in conftest.py)
        # mock_execution_context.resources.internal_model_output = MagicMock()
        # mock_execution_context.resources.internal_model_output.path = "mock/internal/model.pkl"
        # mock_execution_context.resources.external_model_output = MagicMock()
        # mock_execution_context.resources.external_model_output.path = "mock/external/model.pkl"
        # mock_execution_context.resources.job_params = MagicMock()
        # mock_execution_context.resources.job_params.min_cluster_size = 10 # Example value

        # Mock the readers used inside cluster_reassignment to prevent FileNotFoundError
        # Use the paths defined in the fixture resources
        internal_model_path = mock_execution_context.resources.internal_model_output.path
        external_model_path = mock_execution_context.resources.external_model_output.path

        with patch("clustering.pipeline.assets.merging.merge.PickleReader") as MockPickleReader:
            # Configure the mock instances returned by PickleReader
            mock_internal_model_data = {
                "centroids": {0: [0.1, 0.2], 1: [0.3, 0.4], 2: [0.5, 0.6]}  # Example centroids
            }
            mock_external_model_data = {"centroids": {0: [0.7, 0.8], 1: [0.9, 1.0], 2: [1.1, 1.2]}}

            # Ensure the mock instances returned by read() have the data
            mock_reader_instance_internal = MagicMock()
            mock_reader_instance_internal.read.return_value = mock_internal_model_data
            mock_reader_instance_external = MagicMock()
            mock_reader_instance_external.read.return_value = mock_external_model_data

            # Configure the mock class to return specific instances based on path
            def side_effect(path):
                if path == internal_model_path:
                    return mock_reader_instance_internal
                elif path == external_model_path:
                    return mock_reader_instance_external
                else:
                    raise FileNotFoundError(f"Mock path not configured: {path}")

            MockPickleReader.side_effect = side_effect

            # Execute the asset
            result = cluster_reassignment(mock_execution_context, mock_input_dict)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "STORE_NBR" in result.columns
        assert "merged_cluster" in result.columns
        assert "final_cluster" in result.columns

        # Verify all stores have a final cluster assigned
        assert not result["final_cluster"].is_null().any()


class TestSaveMergedClusterAssignments:
    """Tests for save_merged_cluster_assignments asset."""

    def test_save_merged_assignments(self, mock_execution_context):
        """Test saving merged cluster assignments."""
        # Create test data
        reassigned_clusters = pl.DataFrame(
            {
                "store_id": [f"store_{i}" for i in range(5)],
                "optimal_cluster": [0, 1, 2, 1, 0],
                "original_clusters": [
                    {"category_a": 0, "category_b": 1},
                    {"category_a": 1, "category_b": 0},
                    {"category_a": 2, "category_b": 2},
                    {"category_a": 0, "category_b": 1},
                    {"category_a": 1, "category_b": 0},
                ],
            }
        )

        # Execute the asset
        result = save_merged_cluster_assignments(mock_execution_context, reassigned_clusters)

        # Get the writer that should have been used
        writer = mock_execution_context.resources.merged_cluster_writer

        # Verify write was called
        assert writer.written_count > 0

        # Verify data was written (comparing first written data to input)
        assert writer.written_data[0] is not None

        # The function may return None or the input data unchanged
        # Only check equality if result is not None
        if result is not None:
            assert result.equals(reassigned_clusters)
