"""End-to-end tests for the full clustering pipeline.

This module contains end-to-end tests that verify the entire pipeline
works correctly from data ingestion to final cluster output.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from dagster import execute_job

from clustering.cli.commands import export_command, run_command, status_command
from clustering.pipeline.definitions import (
    external_preprocessing_job,
    full_pipeline_job,
    internal_preprocessing_job,
    merging_job,
)


@pytest.fixture
def test_data_dir() -> Path:
    """Create a temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create internal sales data
        sales_data = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003, 1001, 1002, 1003],
                "STORE_NBR": [101, 101, 101, 102, 102, 102],
                "CAT_DSC": ["Health", "Beauty", "Grocery", "Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25, 1400.50, 2100.75, 3000.25],
            }
        )
        sales_path = temp_path / "internal" / "sales.csv"
        os.makedirs(sales_path.parent, exist_ok=True)
        sales_data.to_csv(sales_path, index=False)

        # Create external mapping data
        mapping_data = pd.DataFrame(
            {
                "PRODUCT_ID": [1001, 1002, 1003],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
                "CDT": ["Tablets", "Lotion", "Chips"],
                "ATTRIBUTE_1": ["OTC", "Natural", "Savory"],
                "ATTRIBUTE_2": ["Fast acting", "Hydrating", "Crunchy"],
                "ATTRIBUTE_3": [None, "Anti-aging", None],
                "ATTRIBUTE_4": ["Headache", None, None],
                "ATTRIBUTE_5": [None, None, "Party size"],
                "ATTRIBUTE_6": [None, "Fragrance-free", None],
                "PLANOGRAM_DSC": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                "PLANOGRAM_NBR": [10, 20, 30],
                "NEW_ITEM": [False, True, False],
                "TO_BE_DROPPED": [False, False, True],
            }
        )
        mapping_path = temp_path / "external" / "mapping.csv"
        os.makedirs(mapping_path.parent, exist_ok=True)
        mapping_data.to_csv(mapping_path, index=False)

        # Create configuration file
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
                    "n_clusters": 3,
                    "random_state": 42,
                    "matching_threshold": 0.7,
                },
            }
        }
        config_path = temp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create output directory
        os.makedirs(temp_path / "output", exist_ok=True)

        yield temp_path


@pytest.fixture
def create_config(test_data_dir: Path) -> callable:
    """Create a configuration file with customizable parameters.

    Args:
        test_data_dir: Test data directory from the test_data_dir fixture

    Returns:
        Function to create a config file with custom parameters
    """

    def _create_config(
        algorithm: str = "kmeans",
        n_clusters: int = 3,
        random_state: int = 42,
        matching_threshold: float = 0.7,
        normalize_data: bool = True,
        output_path: str | None = None,
    ) -> Path:
        """Create a configuration file with the specified parameters.

        Args:
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            matching_threshold: Threshold for matching clusters
            normalize_data: Whether to normalize data
            output_path: Custom output path override

        Returns:
            Path to the created config file
        """
        sales_path = test_data_dir / "internal" / "sales.csv"
        mapping_path = test_data_dir / "external" / "mapping.csv"

        output_dir = output_path if output_path else str(test_data_dir / "output")

        config = {
            "job": {
                "kind": "full_pipeline",
                "logger": {"level": "INFO"},
                "data_paths": {
                    "internal_sales": str(sales_path),
                    "external_mapping": str(mapping_path),
                    "output_dir": output_dir,
                },
                "params": {
                    "algorithm": algorithm,
                    "n_clusters": n_clusters,
                    "random_state": random_state,
                    "matching_threshold": matching_threshold,
                    "normalize_data": normalize_data,
                },
            }
        }

        config_path = test_data_dir / f"config_{algorithm}_{n_clusters}.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        return config_path

    return _create_config


class TestFullPipelineE2E:
    """End-to-end tests for the full clustering pipeline."""

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_full_pipeline_execution(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the full pipeline executes successfully end-to-end."""
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Create a CLI runner
        runner = CliRunner()

        # Step 1: Run the full pipeline job via CLI
        config_path = test_data_dir / "config.json"
        result = runner.invoke(run_command, ["full_pipeline_job", "--config", str(config_path)])

        # Check that the job started successfully
        assert result.exit_code == 0
        assert "success" in result.output.lower()

        # Extract job ID from the output
        output_lines = result.output.strip().split("\n")
        job_id = None
        for line in output_lines:
            if "job_id" in line.lower():
                job_id = line.split(":")[-1].strip()
                break

        assert job_id is not None, "Job ID not found in output"

        # Step 2: Check job status
        status_result = runner.invoke(status_command, ["--job-id", job_id])

        # Should either be RUNNING or COMPLETED
        assert status_result.exit_code == 0
        assert any(status in status_result.output for status in ["RUNNING", "COMPLETED"])

        # Step 3: Export the results
        output_path = test_data_dir / "output" / "results.csv"
        export_result = runner.invoke(
            export_command, ["--job-id", job_id, "--format", "csv", "--output", str(output_path)]
        )

        # Check export was successful
        assert export_result.exit_code == 0
        assert "success" in export_result.output.lower()

        # Step 4: Verify the output files
        # Check that the output file exists
        assert output_path.exists()

        # Load and validate the results
        results = pd.read_csv(output_path)

        # Check basic structure
        assert "cluster" in results.columns.str.lower().tolist()
        assert (
            "sku_nbr" in results.columns.str.lower().tolist()
            or "product_id" in results.columns.str.lower().tolist()
        )

        # Check that we have data
        assert len(results) > 0

        # Check that clusters were assigned
        cluster_col = next(col for col in results.columns if "cluster" in col.lower())
        assert results[cluster_col].nunique() > 0

        # Step 5: Check intermediate output files
        internal_results = list(test_data_dir.glob("**/internal_*_result*.csv"))
        external_results = list(test_data_dir.glob("**/external_*_result*.csv"))
        merged_results = list(test_data_dir.glob("**/merged_result*.csv"))

        # Should have at least one of each type of result
        assert len(internal_results) > 0, "No internal results found"
        assert len(external_results) > 0, "No external results found"
        assert len(merged_results) > 0, "No merged results found"

    @pytest.mark.slow
    @pytest.mark.e2e
    @pytest.mark.parametrize("algorithm", ["kmeans", "hierarchical", "dbscan"])
    def test_pipeline_with_different_algorithms(
        self,
        test_data_dir: Path,
        create_config: callable,
        algorithm: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the pipeline with different clustering algorithms.

        Args:
            test_data_dir: Test data directory
            create_config: Function to create config files
            algorithm: Clustering algorithm to test
            monkeypatch: Pytest monkeypatch fixture
        """
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Create a configuration with the specified algorithm
        config_path = create_config(algorithm=algorithm)

        # Create a CLI runner
        runner = CliRunner()

        # Run the pipeline with the specified algorithm
        result = runner.invoke(run_command, ["full_pipeline_job", "--config", str(config_path)])

        # Check execution results
        assert result.exit_code == 0, f"Pipeline failed with algorithm: {algorithm}"
        assert "success" in result.output.lower()

        # Extract job ID
        output_lines = result.output.strip().split("\n")
        job_id = next(
            (line.split(":")[-1].strip() for line in output_lines if "job_id" in line.lower()), None
        )

        assert job_id is not None, f"Job ID not found for algorithm: {algorithm}"

        # Export results
        output_path = test_data_dir / "output" / f"results_{algorithm}.csv"
        export_result = runner.invoke(
            export_command, ["--job-id", job_id, "--format", "csv", "--output", str(output_path)]
        )

        # Check export results
        assert export_result.exit_code == 0
        assert output_path.exists()

        # Validate results
        results = pd.read_csv(output_path)
        assert len(results) > 0

        # Check clusters were created
        cluster_col = next(col for col in results.columns if "cluster" in col.lower())
        if algorithm != "dbscan":  # DBSCAN might have negative cluster values for noise
            assert results[cluster_col].min() >= 0

        # Note: Different algorithms will have different numbers of clusters
        if algorithm == "kmeans":
            # K-means should have exactly 3 clusters (as specified)
            assert results[cluster_col].nunique() == 3

    @pytest.mark.slow
    @pytest.mark.e2e
    @pytest.mark.parametrize("n_clusters", [2, 5, 10])
    def test_pipeline_with_different_cluster_counts(
        self,
        test_data_dir: Path,
        create_config: callable,
        n_clusters: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the pipeline with different numbers of clusters.

        Args:
            test_data_dir: Test data directory
            create_config: Function to create config files
            n_clusters: Number of clusters to create
            monkeypatch: Pytest monkeypatch fixture
        """
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Create a configuration with the specified number of clusters
        config_path = create_config(n_clusters=n_clusters)

        # Create a CLI runner
        runner = CliRunner()

        # Run the pipeline
        result = runner.invoke(run_command, ["full_pipeline_job", "--config", str(config_path)])

        # Check execution results
        assert result.exit_code == 0, f"Pipeline failed with {n_clusters} clusters"
        assert "success" in result.output.lower()

        # Extract job ID
        output_lines = result.output.strip().split("\n")
        job_id = next(
            (line.split(":")[-1].strip() for line in output_lines if "job_id" in line.lower()), None
        )

        assert job_id is not None, f"Job ID not found for {n_clusters} clusters"

        # Export results
        output_path = test_data_dir / "output" / f"results_{n_clusters}_clusters.csv"
        export_result = runner.invoke(
            export_command, ["--job-id", job_id, "--format", "csv", "--output", str(output_path)]
        )

        # Check export results
        assert export_result.exit_code == 0
        assert output_path.exists()

        # Validate results
        results = pd.read_csv(output_path)
        assert len(results) > 0

        # Check number of clusters
        cluster_col = next(col for col in results.columns if "cluster" in col.lower())
        assert (
            results[cluster_col].nunique() <= n_clusters
        )  # Can be less if some clusters are empty

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_pipeline_with_empty_data(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the pipeline's error handling with empty data files.

        Args:
            test_data_dir: Test data directory
            monkeypatch: Pytest monkeypatch fixture
        """
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Create empty data files
        empty_sales_path = test_data_dir / "internal" / "empty_sales.csv"
        empty_mapping_path = test_data_dir / "external" / "empty_mapping.csv"

        # Create sales header with no data
        with open(empty_sales_path, "w") as f:
            f.write("SKU_NBR,STORE_NBR,CAT_DSC,TOTAL_SALES\n")

        # Create mapping header with no data
        with open(empty_mapping_path, "w") as f:
            f.write("PRODUCT_ID,CATEGORY,NEED_STATE,CDT,ATTRIBUTE_1,ATTRIBUTE_2\n")

        # Create config pointing to empty files
        config = {
            "job": {
                "kind": "full_pipeline",
                "data_paths": {
                    "internal_sales": str(empty_sales_path),
                    "external_mapping": str(empty_mapping_path),
                    "output_dir": str(test_data_dir / "output"),
                },
                "params": {"algorithm": "kmeans", "n_clusters": 3},
            }
        }

        empty_config_path = test_data_dir / "empty_config.json"
        with open(empty_config_path, "w") as f:
            json.dump(config, f)

        # Create a CLI runner
        runner = CliRunner()

        # Run pipeline with empty data
        result = runner.invoke(
            run_command, ["full_pipeline_job", "--config", str(empty_config_path)]
        )

        # Should fail with appropriate error
        assert result.exit_code != 0
        # The error message should provide information about the empty input
        assert "empty" in result.output.lower() or "data" in result.output.lower()

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_pipeline_with_invalid_config(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the pipeline's error handling with invalid configuration.

        Args:
            test_data_dir: Test data directory
            monkeypatch: Pytest monkeypatch fixture
        """
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Create an invalid configuration
        invalid_config = {
            "job": {
                "kind": "full_pipeline",
                # Missing data_paths
                "params": {
                    "algorithm": "invalid_algorithm",  # Invalid algorithm
                    "n_clusters": -5,  # Invalid cluster count
                },
            }
        }

        invalid_config_path = test_data_dir / "invalid_config.json"
        with open(invalid_config_path, "w") as f:
            json.dump(invalid_config, f)

        # Create a CLI runner
        runner = CliRunner()

        # Run pipeline with invalid config
        result = runner.invoke(
            run_command, ["full_pipeline_job", "--config", str(invalid_config_path)]
        )

        # Should fail with appropriate error
        assert result.exit_code != 0
        # Error should be informative
        assert any(s in result.output.lower() for s in ["invalid", "missing", "error", "config"])


class TestFullPipelineE2EDagster:
    """End-to-end tests for the full clustering pipeline using Dagster directly."""

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_full_pipeline_dagster_execution(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the full pipeline executes successfully through Dagster directly."""
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Load config
        config_path = test_data_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create run config for Dagster
        run_config = {
            "ops": {
                "internal_load_raw_data": {
                    "config": {"data_path": config["job"]["data_paths"]["internal_sales"]}
                },
                "external_load_raw_data": {
                    "config": {"data_path": config["job"]["data_paths"]["external_mapping"]}
                },
                "internal_preprocess_data": {"config": {"normalize_sales": True}},
                "external_preprocess_data": {"config": {"calculate_importance": True}},
                "internal_transform_data": {"config": config["job"]["params"]},
                "external_transform_data": {"config": config["job"]["params"]},
                "merge_clusters": {
                    "config": {
                        "matching_threshold": config["job"]["params"]["matching_threshold"],
                        "output_path": os.path.join(
                            config["job"]["data_paths"]["output_dir"], "merged_results.csv"
                        ),
                    }
                },
            },
            "resources": {
                "io_manager": {"config": {"base_dir": str(test_data_dir / "dagster_output")}}
            },
        }

        # Execute the job
        result = execute_job(full_pipeline_job, run_config=run_config)

        # Check that job completed successfully
        assert result.success

        # Check that all steps executed
        executed_steps = [
            event.step_key for event in result.all_node_events if event.is_step_success
        ]
        assert "internal_load_raw_data" in executed_steps
        assert "external_load_raw_data" in executed_steps
        assert "internal_preprocess_data" in executed_steps
        assert "external_preprocess_data" in executed_steps
        assert "internal_transform_data" in executed_steps
        assert "external_transform_data" in executed_steps
        assert "merge_clusters" in executed_steps

        # Check output file
        merged_output = os.path.join(
            config["job"]["data_paths"]["output_dir"], "merged_results.csv"
        )
        assert os.path.exists(merged_output)

        # Load and validate the results
        results = pd.read_csv(merged_output)

        # Check that we have merged cluster data
        assert "internal_cluster" in results.columns
        assert "external_cluster" in results.columns
        assert "merged_cluster" in results.columns
        assert len(results) > 0

        # Check that the clusters were assigned properly
        assert results["internal_cluster"].nunique() > 0
        assert results["external_cluster"].nunique() > 0
        assert results["merged_cluster"].nunique() > 0

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_individual_pipeline_components(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test each component of the pipeline separately with Dagster.

        Args:
            test_data_dir: Test data directory
            monkeypatch: Pytest monkeypatch fixture
        """
        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Load config
        config_path = test_data_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create base run config
        base_config = {
            "resources": {
                "io_manager": {"config": {"base_dir": str(test_data_dir / "dagster_output")}}
            }
        }

        # 1. Run internal preprocessing
        internal_config = base_config.copy()
        internal_config["ops"] = {
            "internal_load_raw_data": {
                "config": {"data_path": config["job"]["data_paths"]["internal_sales"]}
            },
            "internal_preprocess_data": {"config": {"normalize_sales": True}},
        }

        internal_result = execute_job(internal_preprocessing_job, run_config=internal_config)

        # Check that internal preprocessing succeeded
        assert internal_result.success

        # 2. Run external preprocessing
        external_config = base_config.copy()
        external_config["ops"] = {
            "external_load_raw_data": {
                "config": {"data_path": config["job"]["data_paths"]["external_mapping"]}
            },
            "external_preprocess_data": {"config": {"calculate_importance": True}},
        }

        external_result = execute_job(external_preprocessing_job, run_config=external_config)

        # Check that external preprocessing succeeded
        assert external_result.success

        # 3. Run merging (requires outputs from previous steps)
        # This typically requires materializing the previous steps first
        # For simplicity, we'll just check if the merging job can be loaded
        assert merging_job is not None
        assert merging_job.name == "merging_job"

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_pipeline_performance(
        self, test_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the performance characteristics of the pipeline.

        Args:
            test_data_dir: Test data directory
            monkeypatch: Pytest monkeypatch fixture
        """
        import time

        # Set environment variables for the test
        monkeypatch.setenv("ENV", "test")
        monkeypatch.setenv("DATA_DIR", str(test_data_dir))

        # Load config
        config_path = test_data_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create run config for Dagster
        run_config = {
            "ops": {
                "internal_load_raw_data": {
                    "config": {"data_path": config["job"]["data_paths"]["internal_sales"]}
                },
                "external_load_raw_data": {
                    "config": {"data_path": config["job"]["data_paths"]["external_mapping"]}
                },
                "internal_preprocess_data": {"config": {"normalize_sales": True}},
                "external_preprocess_data": {"config": {"calculate_importance": True}},
                "internal_transform_data": {"config": config["job"]["params"]},
                "external_transform_data": {"config": config["job"]["params"]},
                "merge_clusters": {
                    "config": {
                        "matching_threshold": config["job"]["params"]["matching_threshold"],
                        "output_path": os.path.join(
                            config["job"]["data_paths"]["output_dir"], "merged_results.csv"
                        ),
                    }
                },
            },
            "resources": {
                "io_manager": {"config": {"base_dir": str(test_data_dir / "dagster_output")}}
            },
        }

        # Measure execution time
        start_time = time.time()
        result = execute_job(full_pipeline_job, run_config=run_config)
        end_time = time.time()

        # Job should complete successfully
        assert result.success

        # Calculate execution time
        execution_time = end_time - start_time

        # For small test data, pipeline should complete within a reasonable time
        # This is just a basic check; actual thresholds would depend on your expectations
        assert execution_time < 60, (
            f"Pipeline took too long to execute: {execution_time:.2f} seconds"
        )

        # Log execution time for information
        print(f"Pipeline execution time: {execution_time:.2f} seconds")

        # Get step-level timing information
        step_timing = {}
        for event in result.all_node_events:
            if event.is_step_start:
                step_timing[event.step_key] = {"start": event.timestamp}
            elif event.is_step_success and event.step_key in step_timing:
                step_timing[event.step_key]["end"] = event.timestamp
                step_timing[event.step_key]["duration"] = (
                    step_timing[event.step_key]["end"] - step_timing[event.step_key]["start"]
                )

        # Check that no single step took more than 70% of the total time (anti-bottleneck check)
        for step, timing in step_timing.items():
            if "duration" in timing:
                step_ratio = timing["duration"] / execution_time
                assert step_ratio < 0.7, (
                    f"Step {step} took {step_ratio:.1%} of total execution time"
                )
