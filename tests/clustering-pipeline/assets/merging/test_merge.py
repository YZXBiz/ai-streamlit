"""Tests for cluster merging assets."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import dagster as dg
from dagster import ResourceDefinition

from clustering.pipeline.assets.merging.merge import (
    cluster_reassignment,
    merged_clusters,
    optimized_merged_clusters,
    save_merged_cluster_assignments,
)


@pytest.fixture
def mock_merge_context() -> dg.AssetExecutionContext:
    """Creates a specialized mock context for merging tests with reader resources.

    Returns:
        Asset execution context configured for merge tests
    """

    # Define the necessary mock classes inline
    class MockConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Define the MockReader class directly in the fixture
    class MockReader:
        def __init__(self, data=None):
            self.data = data or {}
            self.path = "mock/path/reader.pkl"

        def read(self):
            return self.data

    # Define a MockWriter class for resources that need write capability
    class MockWriter:
        def __init__(self):
            self.written_data = []
            self.written_count = 0
            self.path = "mock/path/writer.pkl"

        def write(self, data, **kwargs):
            self.written_data.append(data)
            self.written_count += 1

    # Create a new context with MockReaders for the clusters resources
    mock_config = MockConfig(remove_null_threshold=0.5, min_cluster_size=10)

    # Define all the needed resources
    resource_defs = {
        "config": dg.ResourceDefinition.hardcoded_resource(mock_config),
        "internal_cluster_assignments": dg.ResourceDefinition.hardcoded_resource(MockReader()),
        "external_cluster_assignments": dg.ResourceDefinition.hardcoded_resource(MockReader()),
        "internal_model_output": dg.ResourceDefinition.hardcoded_resource(MockWriter()),
        "external_model_output": dg.ResourceDefinition.hardcoded_resource(MockWriter()),
        "job_params": dg.ResourceDefinition.hardcoded_resource(MockConfig(min_cluster_size=10)),
        "merged_cluster_assignments": dg.ResourceDefinition.hardcoded_resource(MockWriter()),
        "merged_cluster_writer": dg.ResourceDefinition.hardcoded_resource(MockWriter()),
    }

    with dg.build_asset_context(resources=resource_defs) as context:
        yield context


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

    @patch("clustering.pipeline.assets.merging.merge.PickleReader")
    @patch("os.path.exists")
    def test_merge_clusters(
        self,
        mock_exists,
        mock_pickle_reader,
        mock_merge_context,
        sample_internal_assignments,
        sample_external_assignments,
    ):
        """Test merging internal and external cluster assignments."""
        # Create new sample assignments with STORE_NBR column
        internal_df = sample_internal_assignments["category_a"].rename({"store_id": "STORE_NBR"})
        external_df = sample_external_assignments["category_b"].rename({"store_id": "STORE_NBR"})

        internal_assignments_modified = {"category_a": internal_df}
        external_assignments_modified = {"category_b": external_df}

        # Set the mock data on the resources directly - this is the key change
        mock_merge_context.resources.internal_cluster_assignments.data = (
            internal_assignments_modified
        )
        mock_merge_context.resources.external_cluster_assignments.data = (
            external_assignments_modified
        )

        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Execute the asset
        result = merged_clusters(mock_merge_context)

        # Verify merged results
        # Check that all stores from both sources are included
        assert "STORE_NBR" in result.columns
        assert result.height == 5  # We should have all 5 stores from the sample data

        # Verify we have cluster columns for both sources
        # Exact column names depend on implementation
        assert any("cluster" in col for col in result.columns)

    @patch("clustering.pipeline.assets.merging.merge.PickleReader")
    @patch("os.path.exists")
    def test_merge_with_missing_stores(self, mock_exists, mock_pickle_reader, mock_merge_context):
        """Test merging when stores don't perfectly overlap."""
        # Create assignments with different store sets
        internal_assignments = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["store_1", "store_2", "store_3"],
                    "cluster": [1, 2, 0],
                    "distance": [0.2, 0.15, 0.18],
                }
            ),
        }

        external_assignments = {
            "category_b": pl.DataFrame(
                {
                    "STORE_NBR": ["store_2", "store_3", "store_4"],
                    "cluster": [2, 1, 0],
                    "distance": [0.14, 0.17, 0.21],
                }
            ),
        }

        # Set the mock data on the resources directly
        mock_merge_context.resources.internal_cluster_assignments.data = internal_assignments
        mock_merge_context.resources.external_cluster_assignments.data = external_assignments

        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Execute the asset
        result = merged_clusters(mock_merge_context)

        # Verify the merge produced a result with store numbers
        assert "STORE_NBR" in result.columns
        assert result.height > 0

        # We should have the stores that appear in both datasets
        store_nbrs = set(result["STORE_NBR"].to_list())
        assert "store_2" in store_nbrs
        assert "store_3" in store_nbrs
        assert len(store_nbrs) == 2, f"Expected 2 stores, got {len(store_nbrs)}: {store_nbrs}"

    @patch("clustering.pipeline.assets.merging.merge.PickleReader")
    @patch("os.path.exists")
    def test_with_duplicate_stores(self, mock_exists, mock_pickle_reader, mock_merge_context):
        """Test merging when there are duplicate stores."""
        # Create assignments with duplicate stores
        internal_assignments = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["store_1", "store_1", "store_2"],  # Duplicate store
                    "cluster": [1, 1, 2],
                    "distance": [0.2, 0.2, 0.15],
                }
            ),
        }

        external_assignments = {
            "category_b": pl.DataFrame(
                {
                    "STORE_NBR": ["store_1", "store_2", "store_2"],  # Duplicate store
                    "cluster": [0, 2, 2],
                    "distance": [0.14, 0.17, 0.17],
                }
            ),
        }

        # Set the mock data on the resources directly
        mock_merge_context.resources.internal_cluster_assignments.data = internal_assignments
        mock_merge_context.resources.external_cluster_assignments.data = external_assignments

        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Execute the asset
        result = merged_clusters(mock_merge_context)

        # Verify the result has the right number of unique stores
        unique_stores = result["STORE_NBR"].n_unique()
        assert unique_stores == 2, f"Expected 2, got {unique_stores}"

        # Verify that duplicates were handled by join
        store_1_rows = result.filter(pl.col("STORE_NBR") == "store_1").height
        store_2_rows = result.filter(pl.col("STORE_NBR") == "store_2").height

        # There should be at least 1 row for each store
        assert store_1_rows >= 1, f"Expected at least 1 row for store_1, got {store_1_rows}"
        assert store_2_rows >= 1, f"Expected at least 1 row for store_2, got {store_2_rows}"

    @patch("clustering.pipeline.assets.merging.merge.PickleReader")
    @patch("os.path.exists")
    def test_with_high_null_columns(self, mock_exists, mock_pickle_reader, mock_merge_context):
        """Test merging with high-null columns that should be dropped."""
        # Create a dataset where one category has many nulls
        internal_assignments = {
            "category_a": pl.DataFrame(
                {
                    "STORE_NBR": ["store_1", "store_2", "store_3"],
                    "cluster": [1, 2, 0],
                    "distance": [0.2, 0.15, 0.18],
                    "mostly_null": [None, None, "value"],  # 2/3 null, should be dropped
                }
            ),
        }

        external_assignments = {
            "category_b": pl.DataFrame(
                {
                    "STORE_NBR": ["store_1", "store_2", "store_3"],
                    "cluster": [0, 1, 2],
                    "distance": [0.14, 0.17, 0.21],
                }
            ),
        }

        # Set the mock data on the resources directly
        mock_merge_context.resources.internal_cluster_assignments.data = internal_assignments
        mock_merge_context.resources.external_cluster_assignments.data = external_assignments

        # Ensure os.path.exists returns True for mock paths
        mock_exists.return_value = True

        # Make sure the config has the null threshold set
        if not hasattr(mock_merge_context.resources.config, "remove_null_threshold"):
            mock_merge_context.resources.config.remove_null_threshold = 0.5

        # Execute the asset
        result = merged_clusters(mock_merge_context)

        # Verify mostly_null column was dropped due to high null percentage
        assert "mostly_null" not in result.columns, "High-null column should have been dropped"

        # Verify other columns were preserved
        assert "STORE_NBR" in result.columns
        assert any("cluster" in col for col in result.columns)


class TestOptimizedMergedClusters:
    """Tests for optimized_merged_clusters asset."""

    def test_optimize_clusters(self, mock_merge_context, sample_merged_clusters):
        """Test identification of small and large clusters based on size."""
        # Set up a merged data with known cluster sizes
        merged_with_clusters = sample_merged_clusters.with_columns(
            (pl.col("category_a_cluster") + pl.col("category_b_cluster") * 10).alias(
                "merged_cluster"
            )
        )

        # Execute the asset
        result = optimized_merged_clusters(
            mock_merge_context,
            merged_with_clusters,
            {"clusters": {"merged_cluster": [0, 1, 2], "count": [5, 15, 30]}},
        )

        # Verify the result structure
        assert isinstance(result, dict)
        assert "small_clusters" in result
        assert "large_clusters" in result
        assert "merged_data" in result

        # Verify small and large classifications based on min_cluster_size
        small_df = result["small_clusters"]
        large_df = result["large_clusters"]
        assert isinstance(small_df, pl.DataFrame)
        assert isinstance(large_df, pl.DataFrame)

        # Small clusters should have count < min_cluster_size (10)
        assert (small_df["count"] < 10).all()
        # Large clusters should have count >= min_cluster_size (10)
        assert (large_df["count"] >= 10).all()

        # Ensure original data is preserved
        assert result["merged_data"] is merged_with_clusters


class TestClusterReassignment:
    """Tests for cluster_reassignment asset."""

    @patch("clustering.pipeline.assets.merging.merge.PickleReader")
    def test_reassign_clusters(
        self, mock_pickle_reader, mock_merge_context, sample_merged_clusters
    ):
        """Test reassigning small clusters to nearest large clusters."""
        # Configure mock PickleReader to return model data
        mock_reader_instance = MagicMock()
        mock_reader_instance.read.return_value = {"cluster_centers": [[0.1, 0.2], [0.6, 0.7]]}
        mock_pickle_reader.return_value = mock_reader_instance

        # Create test data for optimized_merged_clusters
        merged_with_cluster = sample_merged_clusters.rename({"store_id": "STORE_NBR"}).with_columns(
            pl.col("category_a_cluster").alias("merged_cluster")
        )

        small_clusters = pl.DataFrame(
            {"merged_cluster": [0], "count": [5]}  # One small cluster
        )

        large_clusters = pl.DataFrame(
            {"merged_cluster": [1, 2], "count": [20, 15]}  # Two large clusters
        )

        test_data = {
            "small_clusters": small_clusters,
            "large_clusters": large_clusters,
            "merged_data": merged_with_cluster,
        }

        # Execute the asset
        result = cluster_reassignment(mock_merge_context, test_data)

        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "STORE_NBR" in result.columns  # Store ID column
        assert "final_cluster" in result.columns  # Final assigned cluster

        # Check all stores have a final cluster assigned
        assert result.filter(pl.col("final_cluster").is_null()).height == 0


class TestSaveMergedClusterAssignments:
    """Tests for save_merged_cluster_assignments asset."""

    def test_save_merged_assignments(self, mock_merge_context):
        """Test saving cluster assignments to persistent storage."""
        # Create test data for cluster_reassignment input
        test_data = pl.DataFrame(
            {
                "STORE_NBR": ["store_1", "store_2", "store_3"],
                "merged_cluster": [0, 1, 2],
                "final_cluster": [1, 1, 2],
            }
        )

        # Access the writer before running the test to check if something is written
        writer = mock_merge_context.resources.merged_cluster_assignments
        written_data_before = len(writer.written_data) if hasattr(writer, "written_data") else 0

        # Execute the asset
        result = save_merged_cluster_assignments(mock_merge_context, test_data)

        # Verify the result is None (nothing is returned)
        assert result is None

        # Verify data was written (if writer supports tracking)
        if hasattr(writer, "written_data"):
            assert writer.written_count > written_data_before
