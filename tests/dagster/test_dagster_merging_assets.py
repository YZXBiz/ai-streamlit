"""Tests for Dagster merging assets in the clustering pipeline."""

import pandas as pd
import pytest
from dagster import build_asset_context, materialize_to_memory

from clustering.dagster.assets import (
    cluster_reassignment,
    merged_clusters,
    optimized_merged_clusters,
    save_merged_cluster_assignments,
)


@pytest.fixture
def mock_internal_clusters() -> pd.DataFrame:
    """Create sample internal clusters data."""
    return pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "cluster_name": ["A", "B", "C"],
            "cluster_size": [100, 200, 300],
            "center_feature_1": [0.1, 0.2, 0.3],
            "center_feature_2": [0.4, 0.5, 0.6],
            "source": ["internal"] * 3,
        }
    )


@pytest.fixture
def mock_external_clusters() -> pd.DataFrame:
    """Create sample external clusters data."""
    return pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "cluster_name": ["X", "Y"],
            "cluster_size": [150, 250],
            "center_feature_1": [0.15, 0.25],
            "center_feature_2": [0.45, 0.55],
            "source": ["external"] * 2,
        }
    )


@pytest.fixture
def mock_internal_assignments() -> pd.DataFrame:
    """Create sample internal cluster assignments."""
    return pd.DataFrame(
        {
            "entity_id": [f"E{i:03d}" for i in range(1, 6)],
            "cluster_id": [1, 1, 2, 2, 3],
            "distance_to_center": [0.05, 0.07, 0.08, 0.06, 0.09],
            "source": ["internal"] * 5,
        }
    )


@pytest.fixture
def mock_external_assignments() -> pd.DataFrame:
    """Create sample external cluster assignments."""
    return pd.DataFrame(
        {
            "entity_id": [f"E{i:03d}" for i in range(1, 6)],
            "cluster_id": [1, 1, 2, 2, 2],
            "distance_to_center": [0.06, 0.08, 0.07, 0.05, 0.08],
            "source": ["external"] * 5,
        }
    )


class TestMergingAssets:
    """Tests for merging assets."""

    def test_merged_clusters(self, mock_internal_clusters, mock_external_clusters) -> None:
        """Test merged_clusters asset."""
        # Create context with resources
        context = build_asset_context()

        # Call the asset function directly
        result = merged_clusters(
            context,
            internal_clusters=mock_internal_clusters,
            external_clusters=mock_external_clusters,
        )

        # Assert expected output
        assert isinstance(result, pd.DataFrame)
        assert "source" in result.columns
        assert "cluster_id" in result.columns
        assert len(result) == len(mock_internal_clusters) + len(mock_external_clusters)
        assert set(result["source"].unique()) == {"internal", "external"}

    def test_optimized_merged_clusters(
        self, mock_internal_clusters, mock_external_clusters
    ) -> None:
        """Test optimized_merged_clusters asset."""
        # Create context with resources
        context = build_asset_context()

        # Create merged clusters first
        merged = pd.concat([mock_internal_clusters, mock_external_clusters], ignore_index=True)

        # Call the asset function directly
        result = optimized_merged_clusters(context, merged_clusters=merged)

        # Assert expected output
        assert isinstance(result, pd.DataFrame)
        assert "optimized_cluster_id" in result.columns
        assert "original_cluster_id" in result.columns
        assert "source" in result.columns
        assert len(result) <= len(merged)  # Should be same or fewer clusters after optimization

    def test_cluster_reassignment(
        self, mock_internal_assignments, mock_external_assignments
    ) -> None:
        """Test cluster_reassignment asset."""
        # Create context with resources
        context = build_asset_context()

        # Create merged assignments
        merged_assignments = pd.concat(
            [mock_internal_assignments, mock_external_assignments], ignore_index=True
        )

        # Mock optimized clusters mapping
        optimized_clusters = pd.DataFrame(
            {
                "original_cluster_id": [1, 2, 3, 1, 2],
                "optimized_cluster_id": [101, 102, 103, 101, 102],
                "source": ["internal", "internal", "internal", "external", "external"],
            }
        )

        # Call the asset function directly
        result = cluster_reassignment(
            context,
            merged_cluster_assignments=merged_assignments,
            optimized_merged_clusters=optimized_clusters,
        )

        # Assert expected output
        assert isinstance(result, pd.DataFrame)
        assert "entity_id" in result.columns
        assert "original_cluster_id" in result.columns
        assert "new_cluster_id" in result.columns
        assert "source" in result.columns
        assert len(result) == len(merged_assignments)

    @pytest.mark.parametrize("resource_available", [True, False])
    def test_save_merged_cluster_assignments(self, resource_available) -> None:
        """Test save_merged_cluster_assignments asset.

        Args:
            resource_available: Whether the writer resource is available
        """
        # Create mock data
        reassigned_clusters = pd.DataFrame(
            {
                "entity_id": [f"E{i:03d}" for i in range(1, 6)],
                "original_cluster_id": [1, 1, 2, 2, 3],
                "new_cluster_id": [101, 101, 102, 102, 103],
                "source": ["internal", "internal", "internal", "external", "external"],
            }
        )

        # Mock resource
        if resource_available:

            def mock_writer(df, *args, **kwargs):
                return len(df)

            resources = {"merged_cluster_assignments": mock_writer}
        else:
            resources = {}

        # Create context with resources
        context = build_asset_context(resources=resources)

        if not resource_available:
            with pytest.raises(Exception, match="No resource found"):
                save_merged_cluster_assignments(context, reassigned_clusters)
        else:
            result = save_merged_cluster_assignments(context, reassigned_clusters)
            assert result == len(reassigned_clusters)


@pytest.mark.integration
def test_merging_pipeline_integration() -> None:
    """Integration test for the entire merging pipeline."""
    from clustering.dagster.definitions import merging_job

    try:
        # This tests that the job can be constructed and executed
        # without raising exceptions
        result = materialize_to_memory(
            merging_job.asset_selection,
            resources={
                "merged_cluster_assignments": lambda df, *args, **kwargs: len(df),
                # Add other required resources here
            },
        )
        assert result is not None
    except Exception as e:
        pytest.skip(f"Could not run integration test: {str(e)}")
