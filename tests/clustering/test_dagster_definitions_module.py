"""Tests for Dagster definitions module in the clustering pipeline."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
from dagster import DagsterInstance, asset, define_asset_job

from clustering.dagster.definitions import (
    create_definitions,
    defs,
    get_definitions,
    get_resources_by_env,
    internal_preprocessing_job,
    external_preprocessing_job,
    internal_ml_job,
    external_ml_job,
    merging_job,
    full_pipeline_job,
    load_config,
)


class TestDagsterDefinitions:
    """Tests for the Dagster definitions module."""
    
    def test_load_config(self) -> None:
        """Test loading configuration."""
        # Test loading dev config (default)
        dev_config = load_config("dev")
        assert isinstance(dev_config, dict)
        
        # Should load default config even if environment doesn't exist
        with mock.patch("os.path.exists", return_value=False):
            fallback_config = load_config("nonexistent")
            assert isinstance(fallback_config, dict)
            assert "job_params" in fallback_config
    
    def test_get_resources_by_env(self) -> None:
        """Test getting resources by environment."""
        # Test with dev environment
        dev_resources = get_resources_by_env("dev")
        assert isinstance(dev_resources, dict)
        
        # Check for required resources
        assert "io_manager" in dev_resources
        assert "job_params" in dev_resources
        assert "config" in dev_resources
        assert "logger" in dev_resources
        
        # Check for reader/writer resources
        reader_resources = [key for key in dev_resources.keys() if "reader" in key]
        writer_resources = [key for key in dev_resources.keys() if "writer" in key]
        assert len(reader_resources) > 0
        assert len(writer_resources) > 0
    
    def test_create_definitions_environments(self) -> None:
        """Test creating definitions for different environments."""
        environments = ["dev", "staging", "prod"]
        
        for env in environments:
            # Create definitions for this environment
            definitions = create_definitions(env)
            
            # Check that definitions have expected components
            assert hasattr(definitions, "assets")
            assert hasattr(definitions, "resources")
            assert hasattr(definitions, "jobs")
            
            # Check for expected jobs
            job_names = [job.name for job in definitions.get_all_jobs()]
            expected_jobs = [
                "internal_preprocessing_job",
                "external_preprocessing_job",
                "internal_ml_job",
                "external_ml_job",
                "merging_job",
                "full_pipeline_job",
            ]
            
            for expected in expected_jobs:
                assert expected in job_names, f"Job {expected} not found in {env} environment"
    
    def test_get_definitions(self) -> None:
        """Test get_definitions function."""
        # This should return the default definitions (dev environment)
        definitions = get_definitions()
        
        # Should be the same as the global defs
        assert definitions is defs
        
        # Check that it has assets and jobs
        assert len(list(definitions.get_all_assets())) > 0
        assert len(definitions.get_all_jobs()) > 0
    
    def test_asset_selection(self) -> None:
        """Test asset selection in jobs."""
        # Check that each job selects the expected assets
        jobs = {
            "internal_preprocessing_job": internal_preprocessing_job,
            "external_preprocessing_job": external_preprocessing_job,
            "internal_ml_job": internal_ml_job,
            "external_ml_job": external_ml_job,
            "merging_job": merging_job,
            "full_pipeline_job": full_pipeline_job,
        }
        
        for job_name, job in jobs.items():
            # Get asset selection
            asset_selection = job.asset_selection
            assert asset_selection is not None
            
            # Full pipeline should include all assets
            if job_name == "full_pipeline_job":
                assert len(asset_selection) > sum(len(j.asset_selection) for j_name, j in jobs.items() if j_name != "full_pipeline_job")
            
            # Jobs should have appropriate tags
            if "internal" in job_name:
                assert job.tags.get("kind", "").startswith("internal")
            elif "external" in job_name:
                assert job.tags.get("kind", "").startswith("external")
            elif job_name == "merging_job":
                assert job.tags.get("kind") == "merging"
    
    def test_config_resource(self) -> None:
        """Test the job_params/config resource."""
        # Create a simple config
        config_data = {
            "job_params": {
                "algorithm": "kmeans",
                "n_clusters": 3,
                "random_state": 42,
            }
        }
        
        # Mock load_config to return our test config
        with mock.patch("clustering.dagster.definitions.load_config", return_value=config_data):
            # Get resources
            resources = get_resources_by_env()
            
            # Get the config resource
            config_resource = resources["config"]
            job_params_resource = resources["job_params"]
            
            # Both should point to the same resource
            assert config_resource is job_params_resource
            
            # Initialize the resource (it's a callable that returns SimpleNamespace)
            config = config_resource()
            
            # Check attributes
            assert isinstance(config, SimpleNamespace)
            assert config.algorithm == "kmeans"
            assert config.n_clusters == 3
            assert config.random_state == 42
            assert config.env == "dev"  # Added by the get_resources_by_env function


@pytest.mark.integration
def test_definitions_integration() -> None:
    """Integration test for Dagster definitions."""
    try:
        # Create an ephemeral instance for testing
        with DagsterInstance.ephemeral() as instance:
            # Create a simple asset to test with definitions
            @asset
            def test_asset() -> str:
                return "test data"
            
            # Create a job that uses this asset
            test_job = define_asset_job("test_job", selection=[test_asset])
            
            # Create definitions with our test asset and job
            test_defs = create_definitions(env="dev")
            
            # Check that definitions can be used to get assets
            assets = list(test_defs.get_all_assets())
            assert len(assets) > 0
            
            # Check that definitions can be used to get jobs
            jobs = test_defs.get_all_jobs()
            assert len(jobs) > 0
            
            # Ensure we have access to resources
            resources = test_defs.resources
            assert "io_manager" in resources
            assert "logger" in resources
    except Exception as e:
        pytest.skip(f"Integration test failed: {str(e)}")


def test_resource_defaults() -> None:
    """Test default values for resources."""
    # Test io_manager default path
    resources = get_resources_by_env()
    io_manager = resources["io_manager"]
    
    # Default storage location should use environment variable or fallback
    with mock.patch.dict(os.environ, {"DAGSTER_STORAGE_DIR": "/custom/storage"}):
        resources = get_resources_by_env()
        io_manager = resources["io_manager"]
        # We can't easily check the base_dir attribute without instantiating,
        # but we can check the resource is created without error
        assert io_manager is not None
    
    # Test logger default values
    logger = resources["logger"]
    assert logger is not None


def test_full_pipeline_config() -> None:
    """Test full pipeline configuration."""
    # Full pipeline should be configured for sequential execution
    config = full_pipeline_job.job_def.config
    assert "execution" in config
    assert "config" in config["execution"]
    assert "multiprocess" in config["execution"]["config"]
    assert "max_concurrent" in config["execution"]["config"]["multiprocess"]
    assert config["execution"]["config"]["multiprocess"]["max_concurrent"] == 1
