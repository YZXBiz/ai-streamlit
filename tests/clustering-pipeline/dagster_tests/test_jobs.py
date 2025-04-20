"""Tests for Dagster jobs in the pipeline package."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dagster import (
    AssetIn,
    AssetKey,
    DagsterInstance,
    RunRequest,
    ScheduleDefinition,
    asset,
    build_schedule_context,
    build_sensor_context,
    define_asset_job,
    job,
    materialize_to_memory,
    mem_io_manager,
    op,
    sensor,
)
from dagster._core.definitions.definitions_class import Definitions
from dagster._core.execution.execute_in_process_result import ExecuteInProcessResult

from clustering.pipeline.definitions import (
    external_preprocessing_job,
    full_pipeline_job,
    internal_preprocessing_job,
    merging_job,
)


@pytest.fixture
def run_dagster_job() -> Callable[..., ExecuteInProcessResult]:
    """Create a fixture for running a Dagster job.

    Returns:
        A function that runs a Dagster job.
    """

    def _run_job(
        job_def: Any, run_config: dict[str, Any] = None, instance: DagsterInstance = None
    ) -> ExecuteInProcessResult:
        # In modern Dagster, most jobs should be executed using .execute_in_process
        return job_def.execute_in_process(run_config=run_config or {}, instance=instance)

    return _run_job


@pytest.fixture
def in_memory_dagster_instance() -> DagsterInstance:
    """Create an in-memory Dagster instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        instance = DagsterInstance.ephemeral(tempdir=temp_dir)
        yield instance


@pytest.fixture
def test_job():
    """Create a simple test job for testing."""

    @op
    def test_op_1() -> int:
        return 42

    @op
    def test_op_2(val: int) -> str:
        return f"Value: {val}"

    @job
    def simple_test_job():
        test_op_2(test_op_1())

    return simple_test_job


class TestJobBasics:
    """Basic tests for Dagster jobs."""

    def test_job_execution(self, test_job) -> None:
        """Test that a simple job can execute successfully."""
        # Execute the job directly
        result = test_job.execute_in_process()

        # Check that the job completed successfully
        assert result.success

        # Check that both ops executed and the second op got the correct value
        assert result.output_for_node("test_op_1") == 42
        assert result.output_for_node("test_op_2") == "Value: 42"

    def test_asset_job(self, in_memory_dagster_instance: DagsterInstance) -> None:
        """Test that an asset-based job executes correctly."""

        # Define test assets
        @asset
        def input_asset() -> pd.DataFrame:
            return pd.DataFrame({"value": [1, 2, 3]})

        @asset(ins={"data": AssetIn("input_asset")})
        def output_asset(data: pd.DataFrame) -> pd.DataFrame:
            return data.assign(doubled=data["value"] * 2)

        # Define a job that materializes these assets
        asset_job = define_asset_job(
            name="test_asset_job", selection=[AssetKey("input_asset"), AssetKey("output_asset")]
        )

        # Create definitions with assets and job
        Definitions(
            assets=[input_asset, output_asset],
            jobs=[asset_job],
        )

        # Use the modern materialize_to_memory approach instead
        result = materialize_to_memory(
            [input_asset, output_asset],
            instance=in_memory_dagster_instance,
            resources={"io_manager": mem_io_manager},
        )

        # Check results
        assert result.success
        assert len(result.asset_materializations_for_node("input_asset")) == 1
        assert len(result.asset_materializations_for_node("output_asset")) == 1


class TestInternalPreprocessingJob:
    """Tests for the internal preprocessing job."""

    def test_internal_preprocessing_job_structure(self) -> None:
        """Test the structure of the internal preprocessing job."""
        # Check that the job exists
        assert internal_preprocessing_job is not None

        # Check job name
        assert internal_preprocessing_job.name == "internal_preprocessing_job"

        # For UnresolvedAssetJobDefinition, we can only check the name
        # We can't check selection_data as that's only available after resolution


class TestExternalPreprocessingJob:
    """Tests for the external preprocessing job."""

    def test_external_preprocessing_job_structure(self) -> None:
        """Test the structure of the external preprocessing job."""
        # Check that the job exists
        assert external_preprocessing_job is not None

        # Check job name
        assert external_preprocessing_job.name == "external_preprocessing_job"


class TestMergingJob:
    """Tests for the merging job."""

    def test_merging_job_structure(self) -> None:
        """Test the structure of the merging job."""
        # Check that the job exists
        assert merging_job is not None

        # Check job name
        assert merging_job.name == "merging_job"


class TestFullPipelineJob:
    """Tests for the full pipeline job."""

    def test_full_pipeline_job_structure(self) -> None:
        """Test the structure of the full pipeline job."""
        # Check that the job exists
        assert full_pipeline_job is not None

        # Check job name
        assert full_pipeline_job.name == "full_pipeline_job"


class TestSchedules:
    """Tests for job schedules."""

    def test_daily_schedule_definition(self) -> None:
        """Test that a daily schedule can be defined and triggered."""

        # Create a test job
        @job
        def test_scheduled_job():
            pass

        # Create a daily schedule for the job
        schedule = ScheduleDefinition(
            name="test_daily_schedule",
            cron_schedule="0 0 * * *",  # Daily at midnight
            job=test_scheduled_job,
            execution_timezone="UTC",
        )

        # Check schedule properties
        assert schedule.name == "test_daily_schedule"
        assert schedule.cron_schedule == "0 0 * * *"
        assert schedule.job_name == "test_scheduled_job"

        # Create a proper context for the schedule evaluation
        context = build_schedule_context()

        # Evaluate the schedule with proper context
        run_request = schedule.evaluate_tick(context)
        # In the new API, evaluate_tick returns a ScheduleExecutionData object with a single RunRequest
        assert isinstance(run_request, RunRequest)
        assert run_request.job_name == "test_scheduled_job"


class TestSensors:
    """Tests for job sensors."""

    def test_file_sensor_definition(self) -> None:
        """Test that a file sensor can be defined and triggered."""

        # Create a test job
        @job
        def test_sensed_job():
            pass

        # Mock directory for file detection
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sensor using the modern API
            @sensor(
                name="test_file_sensor",
                job=test_sensed_job,
                minimum_interval_seconds=30,
            )
            def file_sensor(context):
                files = list(Path(temp_dir).glob("*.csv"))
                if files:
                    return RunRequest(
                        run_key=files[0].name,
                        job_name="test_sensed_job",
                        run_config={"path": str(files[0])},
                    )
                return None

            # Create the sensor from the decorated function
            sensor_def = file_sensor

            # Check sensor properties
            assert sensor_def.name == "test_file_sensor"

            # Create proper sensor context
            context = build_sensor_context()

            # No file yet, so no run request
            result = sensor_def(context)
            assert result is None

            # Add a file and check that it creates a run request
            test_file = Path(temp_dir) / "test.csv"
            with open(test_file, "w") as f:
                f.write("a,b,c\n1,2,3\n")

            # Now should return a run request
            result = sensor_def(context)
            assert isinstance(result, RunRequest)
            assert result.job_name == "test_sensed_job"
            assert result.run_key == "test.csv"
