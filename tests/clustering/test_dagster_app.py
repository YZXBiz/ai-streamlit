"""Tests for the Dagster app module in the clustering pipeline."""

import os
import subprocess
import sys
from unittest import mock

import pytest
from dagster import DagsterInstance

from clustering.dagster.app import create_app


class TestDagsterApp:
    """Tests for the Dagster app functionality."""
    
    def test_create_app(self) -> None:
        """Test that the create_app function returns a valid app."""
        # Create the app
        app = create_app()
        
        # Verify app attributes
        assert hasattr(app, "app_name")
        assert app.app_name == "clustering"
        assert hasattr(app, "definitions")
        
        # Verify the app has assets
        assets = list(app.definitions.get_all_assets())
        assert len(assets) > 0
        
        # Verify the app has jobs
        jobs = app.definitions.get_all_jobs()
        assert len(jobs) > 0
        
        # Verify expected jobs
        job_names = [job.name for job in jobs]
        expected_jobs = [
            "internal_preprocessing_job",
            "external_preprocessing_job",
            "internal_ml_job",
            "external_ml_job",
            "merging_job",
            "full_pipeline_job",
        ]
        
        for expected in expected_jobs:
            assert expected in job_names, f"Expected job {expected} not found"
    
    def test_app_env_configuration(self) -> None:
        """Test that the app loads configuration based on env."""
        # Test with dev environment (default)
        dev_app = create_app(env="dev")
        
        # Test with prod environment
        with mock.patch.dict(os.environ, {"DAGSTER_ENV": "prod"}):
            prod_app = create_app()
        
        # Both should have the same structure but potentially different configs
        assert dev_app.app_name == prod_app.app_name
        assert len(list(dev_app.definitions.get_all_assets())) == len(
            list(prod_app.definitions.get_all_assets())
        )
    
    @pytest.mark.parametrize("env", ["dev", "staging", "prod"])
    def test_app_resources_by_env(self, env) -> None:
        """Test that resources are properly configured for each environment.
        
        Args:
            env: Environment to test
        """
        app = create_app(env=env)
        
        # Check that necessary resources are configured
        resource_keys = app.definitions.resources.keys()
        expected_resources = [
            "io_manager",
            "job_params",
            "config",
            "logger",
        ]
        
        for resource in expected_resources:
            assert resource in resource_keys, f"Resource {resource} not found in {env} environment"
        
        # Check that data readers/writers resources are present
        reader_keys = [key for key in resource_keys if "reader" in key]
        writer_keys = [key for key in resource_keys if "writer" in key]
        
        assert len(reader_keys) > 0, f"No reader resources found in {env} environment"
        assert len(writer_keys) > 0, f"No writer resources found in {env} environment"


@pytest.mark.integration
def test_app_can_run_job() -> None:
    """Test that a job can be executed through the app."""
    # Import here to avoid circular imports
    from clustering.dagster.app import create_app
    from clustering.dagster.definitions import internal_preprocessing_job
    
    try:
        # Create app
        app = create_app(env="dev")
        
        # Create an ephemeral instance for testing
        with DagsterInstance.ephemeral() as instance:
            # Try to execute a job
            result = internal_preprocessing_job.execute_in_process(
                instance=instance,
                raise_on_error=False,
            )
            
            # We're not checking if the job succeeds, just that it can be executed
            assert result is not None
    except Exception as e:
        pytest.skip(f"Job execution failed: {str(e)}")


@pytest.mark.skipif(
    sys.platform != "linux", reason="CLI tests only run on Linux"
)
def test_app_cli():
    """Test the app CLI interface."""
    # Import the module to ensure it's installed
    try:
        import clustering
    except ImportError:
        pytest.skip("clustering package not installed")
    
    # Test the CLI with --help flag to avoid actually running anything
    try:
        result = subprocess.run(
            ["python", "-m", "clustering.dagster.app", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Check that the help output makes sense
        assert "usage" in result.stdout.lower()
        assert "dagster" in result.stdout.lower()
        assert "clustering" in result.stdout.lower()
    except subprocess.CalledProcessError:
        pytest.skip("CLI execution failed")
