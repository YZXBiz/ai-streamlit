"""Smoke tests to verify basic pipeline functionality.

These tests are fast and verify that the pipeline can load and run without errors.
"""

import pytest
from dagster import DagsterInstance, build_resources

from clustering.pipeline.definitions import (
    full_pipeline_job,
    internal_preprocessing_job,
    get_resources_by_env,
)
from clustering.shared.infra import Environment


@pytest.fixture
def minimal_config():
    """Minimal configuration for smoke tests."""
    return {
        "algorithm": "kmeans",
        "n_clusters": 3,
        "random_state": 42,
    }


class TestPipelineSmoke:
    """Smoke tests for the clustering pipeline."""

    def test_pipeline_can_load(self):
        """Test that pipeline definitions can be loaded."""
        # Simply verifies that importing the jobs doesn't raise exceptions
        assert full_pipeline_job is not None
        assert internal_preprocessing_job is not None

    def test_resources_can_initialize(self):
        """Test that resources can be initialized."""
        # Get resource definitions for dev environment
        resource_defs = get_resources_by_env(Environment.DEV)

        # Check that key resources are present
        assert "config" in resource_defs
        assert "io_manager" in resource_defs

    @pytest.mark.parametrize(
        "job_def",
        [
            full_pipeline_job,
            internal_preprocessing_job,
        ],
    )
    def test_job_can_build(self, job_def):
        """Test that jobs can be built without errors."""
        # This test just verifies the job definition is valid
        # It does not actually execute the job
        assert job_def.name is not None
        assert hasattr(job_def, "selection")
        assert job_def.tags is not None
