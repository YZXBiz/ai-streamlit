"""Tests for cluster merging assets."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from clustering.pipeline.assets.merging.merge import (
    merged_clusters,
    merged_cluster_assignments,
    optimized_merged_clusters,
    cluster_reassignment,
    save_merged_cluster_assignments,
)


@pytest.fixture
def sample_internal_assignments() -> dict[str, pl.DataFrame]:
    """Create sample internal store cluster assignments."""
    return {
        "category_a": pl.DataFrame({
            "store_id": [f"store_{i}" for i in range(5)],
            "cluster": [0, 1, 2, 0, 1],
            "distance": [0.1, 0.2, 0.15, 0.18, 0.22],
        }),
    }


@pytest.fixture
def sample_external_assignments() -> dict[str, pl.DataFrame]:
    """Create sample external store cluster assignments."""
    return {
        "category_b": pl.DataFrame({
            "store_id": [f"store_{i}" for i in range(5)],
            "cluster": [1, 0, 2, 1, 0],
            "distance": [0.12, 0.19, 0.14, 0.17, 0.21],
        }),
    }


@pytest.fixture
def sample_merged_clusters() -> pl.DataFrame:
    """Create sample merged clusters data."""
    return pl.DataFrame({
        "store_id": [f"store_{i}" for i in range(5)],
        "category_a_cluster": [0, 1, 2, 0, 1],
        "category_b_cluster": [1, 0, 2, 1, 0],
        "category_a_distance": [0.1, 0.2, 0.15, 0.18, 0.22],
        "category_b_distance": [0.12, 0.19, 0.14, 0.17, 0.21],
    })


class TestMergedClusters:
    """Tests for merged_clusters asset."""
    
    def test_merge_clusters(self, mock_execution_context, sample_internal_assignments, sample_external_assignments):
        """Test merging internal and external cluster assignments."""
        # Execute the asset
        result = merged_clusters(
            mock_execution_context,
            sample_internal_assignments,
            sample_external_assignments
        )
        
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
    
    def test_merge_with_missing_stores(self, mock_execution_context):
        """Test merging when stores don't perfectly overlap."""
        # Create assignments with different store sets
        internal_assignments = {
            "category_a": pl.DataFrame({
                "store_id": ["store_1", "store_2", "store_3"],
                "cluster": [1, 2, 0],
                "distance": [0.2, 0.15, 0.18],
            }),
        }
        
        external_assignments = {
            "category_b": pl.DataFrame({
                "store_id": ["store_2", "store_3", "store_4"],
                "cluster": [2, 1, 0],
                "distance": [0.14, 0.17, 0.21],
            }),
        }
        
        # Execute the asset
        result = merged_clusters(
            mock_execution_context,
            internal_assignments,
            external_assignments
        )
        
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
        mock_execution_context.resources.config.category_weights = {"category_a": 0.6, "category_b": 0.4}
        
        # Execute the asset
        result = optimized_merged_clusters(mock_execution_context, sample_merged_clusters)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "category_a_cluster" in result.columns
        assert "category_b_cluster" in result.columns
        assert "optimal_cluster" in result.columns
        assert "score" in result.columns
        
        # Verify all stores have an optimal cluster assigned
        assert not result["optimal_cluster"].is_null().any()
        
        # Verify scores are calculated (not null)
        assert not result["score"].is_null().any()


class TestClusterReassignment:
    """Tests for cluster_reassignment asset."""
    
    def test_reassign_clusters(self, mock_execution_context, sample_merged_clusters):
        """Test reassigning clusters based on optimization."""
        # Create optimized clusters
        optimized_clusters = sample_merged_clusters.with_columns([
            pl.lit(1).alias("optimal_cluster"),
            pl.lit(0.85).alias("score"),
        ])
        
        # Execute the asset
        result = cluster_reassignment(mock_execution_context, optimized_clusters)
        
        # Verify result structure
        assert isinstance(result, pl.DataFrame)
        assert "store_id" in result.columns
        assert "optimal_cluster" in result.columns
        assert "original_clusters" in result.columns
        
        # Verify all stores have a reassigned cluster
        assert not result["optimal_cluster"].is_null().any()
        
        # Verify original clusters are preserved
        for row in result.rows(named=True):
            original = row["original_clusters"]
            assert "category_a" in original
            assert "category_b" in original


class TestSaveMergedClusterAssignments:
    """Tests for save_merged_cluster_assignments asset."""
    
    def test_save_merged_assignments(self, mock_execution_context):
        """Test saving merged cluster assignments."""
        # Create test data
        reassigned_clusters = pl.DataFrame({
            "store_id": [f"store_{i}" for i in range(5)],
            "optimal_cluster": [0, 1, 2, 1, 0],
            "original_clusters": [
                {"category_a": 0, "category_b": 1},
                {"category_a": 1, "category_b": 0},
                {"category_a": 2, "category_b": 2},
                {"category_a": 0, "category_b": 1},
                {"category_a": 1, "category_b": 0},
            ],
        })
        
        # Execute the asset
        result = save_merged_cluster_assignments(mock_execution_context, reassigned_clusters)
        
        # Get the writer that should have been used
        writer = mock_execution_context.resources.merged_cluster_writer
        
        # Verify write was called
        assert writer.written_count > 0
        
        # Verify data was written (comparing first written data to input)
        assert writer.written_data[0] is not None
        
        # Verify the result is passed through unchanged
        assert result.equals(reassigned_clusters) 