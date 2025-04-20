"""Tests for pipeline definitions and asset checks."""

import dagster as dg
import pytest

# Import only what's actually defined in the file
from clustering.pipeline.definitions import (
    defs,
    # Remove asset_checks reference
    # Remove all_assets reference
)


class TestPipelineDefinitions:
    """Tests for the Dagster pipeline definitions."""
    
    def test_repository_definition(self):
        """Test that the repository definition is valid."""
        # Verify the repository definition is a valid Dagster object
        assert isinstance(defs, dg.Definitions)
        
        # Verify it contains the expected components
        assert hasattr(defs, "assets")
        # Remove assertion for asset_checks as they don't exist
        assert hasattr(defs, "resources")
        assert hasattr(defs, "jobs")
    
    def test_assets_defined(self):
        """Test that all required assets are defined."""
        # Get the assets from the repository
        assets = defs.get_asset_defs()
        
        # Verify we have assets
        assert len(assets) > 0
        
        # Verify key assets are included
        asset_keys = [asset.key.to_python_identifier() for asset in assets]
        
        # Internal preprocessing assets
        assert "internal_raw_sales_data" in asset_keys
        assert "internal_product_category_mapping" in asset_keys
        assert "internal_sales_with_categories" in asset_keys
        assert "internal_sales_by_category" in asset_keys
        assert "internal_output_sales_table" in asset_keys
        
        # External preprocessing assets
        assert "external_features_data" in asset_keys
        assert "preprocessed_external_data" in asset_keys
        
        # Internal clustering assets
        assert "internal_fe_raw_data" in asset_keys
        assert "internal_filtered_features" in asset_keys
        assert "internal_normalized_data" in asset_keys
        assert "internal_optimal_cluster_counts" in asset_keys
        assert "internal_train_clustering_models" in asset_keys
        
        # Merging assets
        assert "merged_clusters" in asset_keys
        assert "optimized_merged_clusters" in asset_keys
        assert "save_merged_cluster_assignments" in asset_keys


class TestJobs:
    """Tests for Dagster jobs."""
    
    def test_job_definitions(self):
        """Test that jobs are properly defined."""
        # Get the jobs from the repository
        jobs = defs.get_job_defs()
        
        # Verify we have jobs
        assert len(jobs) > 0
        
        # Verify each job has the expected attributes
        for job in jobs:
            assert isinstance(job, dg.JobDefinition)
            assert job.name is not None
            
            # Get the job's execution plan
            execution_plan = job.get_execution_plan()
            assert execution_plan is not None
            
            # Verify the job has steps
            assert len(execution_plan.step_keys_in_plan) > 0 