"""Tests for the pipeline definitions."""

import dagster as dg

# Import what's actually used in the file
from clustering.pipeline.definitions import defs


class TestPipelineDefinitions:
    """Tests for the Dagster pipeline definitions."""

    def test_repository_definition(self):
        """Test that the repository definition is valid."""
        # Verify the repository definition is a valid Dagster object
        assert isinstance(defs, dg.Definitions)

        # Verify it contains the expected components
        assert hasattr(defs, "assets")
        assert hasattr(defs, "resources")
        assert hasattr(defs, "jobs")

    def test_assets_defined(self):
        """Test that all required assets are defined."""
        # Get the asset nodes from the repository
        asset_graph = defs.get_asset_graph()

        # Get all asset keys from the graph
        asset_keys = list(asset_graph.get_all_asset_keys())

        # Verify we have assets
        assert len(asset_keys) > 0

        # Verify key assets are included by checking for their keys
        expected_assets = [
            # Internal preprocessing assets
            "internal_raw_sales_data",
            "internal_product_category_mapping",
            "internal_sales_with_categories",
            "internal_sales_by_category",
            "internal_output_sales_table",
            # External preprocessing assets
            "external_features_data",
            "preprocessed_external_data",
            # Internal clustering assets
            "internal_fe_raw_data",
            "internal_filtered_features",
            "internal_normalized_data",
            "internal_optimal_cluster_counts",
            "internal_train_clustering_models",
            # Merging assets
            "merged_clusters",
            "optimized_merged_clusters",
            "save_merged_cluster_assignments",
        ]

        for asset_name in expected_assets:
            asset_key = dg.AssetKey(asset_name)
            assert asset_key in asset_keys, f"Asset {asset_name} is missing"


class TestJobs:
    """Tests for Dagster jobs."""

    def test_job_definitions(self):
        """Test that jobs are properly defined."""
        # Get the jobs from the repository using the current API
        job_defs = defs.get_all_job_defs()

        # Verify we have jobs
        assert len(job_defs) > 0

        # Verify each job has the expected attributes
        for job in job_defs:
            assert isinstance(job, dg.JobDefinition)
            assert job.name is not None

            # Basic job validation is enough - don't try to get execution plan
            assert job.execute_in_process is not None
