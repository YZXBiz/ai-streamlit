"""End-to-end tests for the clustering pipeline CLI.

These tests verify that the pipeline can be executed through the CLI
and produces the expected outputs.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from clustering.cli.commands import run_command, status_command, export_command


@pytest.fixture
def minimal_test_data():
    """Create minimal test data files in a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create internal sales data (minimal)
        sales_data = pd.DataFrame({
            "SKU_NBR": [1001, 1002, 1003, 1001, 1002, 1003],
            "STORE_NBR": ["101", "101", "101", "102", "102", "102"],
            "CAT_DSC": ["Health", "Beauty", "Grocery", "Health", "Beauty", "Grocery"],
            "TOTAL_SALES": [1500.0, 2200.0, 3100.0, 1400.0, 2100.0, 3000.0],
        })
        sales_path = temp_path / "internal" / "sales.csv"
        os.makedirs(sales_path.parent, exist_ok=True)
        sales_data.to_csv(sales_path, index=False)
        
        # Create external mapping data (minimal)
        mapping_data = pd.DataFrame({
            "PRODUCT_ID": [1001, 1002, 1003],
            "CATEGORY": ["Health", "Beauty", "Grocery"],
            "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
        })
        mapping_path = temp_path / "external" / "mapping.csv"
        os.makedirs(mapping_path.parent, exist_ok=True)
        mapping_data.to_csv(mapping_path, index=False)
        
        # Create basic config
        config = {
            "job": {
                "kind": "full_pipeline",
                "logger": {"level": "INFO"},
                "data_paths": {
                    "internal_sales": str(sales_path),
                    "external_mapping": str(mapping_path),
                    "output_dir": str(temp_path / "output"),
                },
                "params": {
                    "algorithm": "kmeans",
                    "n_clusters": 2,
                    "random_state": 42,
                },
            }
        }
        config_path = temp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        # Create output directory
        os.makedirs(temp_path / "output", exist_ok=True)
        
        yield temp_path


class TestPipelineCliE2E:
    """End-to-end tests for the pipeline CLI."""
    
    @pytest.mark.e2e
    def test_cli_run_and_status(self, minimal_test_data, monkeypatch):
        """Test that the pipeline can be run via CLI and status checked."""
        # Set environment variables
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(minimal_test_data))
        
        # Create CLI runner
        runner = CliRunner()
        
        # Step 1: Run the pipeline job
        config_path = minimal_test_data / "config.json"
        result = runner.invoke(run_command, ["full_pipeline_job", "--config", str(config_path)])
        
        # Check that job started successfully
        assert result.exit_code == 0
        
        # Extract job ID from output
        job_id = None
        for line in result.output.strip().split("\n"):
            if "job_id" in line.lower():
                job_id = line.split(":")[-1].strip()
                break
                
        assert job_id is not None, "Job ID not found in output"
        
        # Step 2: Check status
        status_result = runner.invoke(status_command, ["--job-id", job_id])
        
        # Status should be either running or completed
        assert status_result.exit_code == 0
        assert any(status in status_result.output for status in ["RUNNING", "COMPLETED", "SUCCESS"])
    
    @pytest.mark.e2e
    def test_export_results(self, minimal_test_data, monkeypatch):
        """Test that results can be exported after pipeline run."""
        # Skip this test if we can't actually run the job in this test environment
        # This is just for illustration
        pytest.skip("This test requires a complete pipeline run")
        
        # Set environment variables
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(minimal_test_data))
        
        # Assume we have a job ID from a successful run
        job_id = "example_job_id"
        
        # Export results
        runner = CliRunner()
        export_path = minimal_test_data / "exported_results.csv"
        export_result = runner.invoke(export_command, [
            "--job-id", job_id,
            "--output", str(export_path)
        ])
        
        # Check export was successful
        assert export_result.exit_code == 0
        assert export_path.exists(), "Export file was not created" 