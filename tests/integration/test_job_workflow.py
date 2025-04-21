"""Integration tests for job workflow."""

import tempfile
from pathlib import Path

import pytest
from dagster import job, op

from clustering.pipeline.definitions import (
    full_pipeline_job,
    internal_preprocessing_job,
    external_preprocessing_job,
    merging_job,
)


class TestJobWorkflow:
    """Tests for job workflow integration."""
    
    def test_job_dependencies(self):
        """Test that job definitions have the correct dependencies."""
        # Test that the jobs can be loaded
        assert full_pipeline_job is not None
        assert internal_preprocessing_job is not None
        assert external_preprocessing_job is not None
        assert merging_job is not None
        
        # Test job names
        assert full_pipeline_job.name == "full_pipeline_job"
        assert internal_preprocessing_job.name == "internal_preprocessing_job"
        assert external_preprocessing_job.name == "external_preprocessing_job"
        assert merging_job.name == "merging_job"
        
    def test_job_tagging(self):
        """Test that jobs have the correct tags."""
        # Check job tags
        assert "kind" in full_pipeline_job.tags
        assert "kind" in internal_preprocessing_job.tags
        assert "kind" in external_preprocessing_job.tags
        assert "kind" in merging_job.tags
        
        # Verify specific tag values
        assert internal_preprocessing_job.tags["kind"] == "internal_preprocessing"
        assert external_preprocessing_job.tags["kind"] == "external_preprocessing"
        assert merging_job.tags["kind"] == "merging"
