"""Tests for CLI commands in the clustering-cli package."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from clustering.cli.commands import (
    export_command,
    list_command,
    run_command,
    status_command,
    validate_command,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture for Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config_file() -> Path:
    """Create a mock configuration file for testing."""
    config = {
        "job": {
            "kind": "test_job",
            "logger": {"level": "INFO"},
            "params": {"algorithm": "kmeans", "n_clusters": 5, "random_state": 42},
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as temp:
        temp_path = Path(temp.name)
        json.dump(config, temp)

    yield temp_path

    if temp_path.exists():
        os.unlink(temp_path)


@pytest.fixture
def sample_data_file() -> Path:
    """Create a sample data file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

        # Create sample data
        data = pd.DataFrame(
            {
                "SKU_NBR": [1001, 1002, 1003],
                "STORE_NBR": [101, 102, 103],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            }
        )

        # Write to CSV
        data.to_csv(temp_path, index=False)

    yield temp_path

    if temp_path.exists():
        os.unlink(temp_path)


class TestRunCommand:
    """Tests for the 'run' command."""

    def test_run_command_with_valid_job(self, cli_runner: CliRunner) -> None:
        """Test running a job with valid parameters."""
        with patch("clustering.cli.commands.run_job") as mock_run_job:
            # Set up the mock
            mock_run_job.return_value = {"status": "success", "job_id": "test-123"}

            # Invoke the command
            result = cli_runner.invoke(
                run_command, ["test_job", "--env", "dev", "--param", "n_clusters=5"]
            )

            # Check the result
            assert result.exit_code == 0
            assert "success" in result.output
            assert "test-123" in result.output

            # Verify mock was called with correct args
            mock_run_job.assert_called_once()
            args, kwargs = mock_run_job.call_args
            assert args[0] == "test_job"
            assert kwargs["env"] == "dev"
            assert kwargs["params"]["n_clusters"] == "5"

    def test_run_command_with_config_file(
        self, cli_runner: CliRunner, mock_config_file: Path
    ) -> None:
        """Test running a job with a config file."""
        with patch("clustering.cli.commands.run_job") as mock_run_job:
            # Set up the mock
            mock_run_job.return_value = {"status": "success", "job_id": "test-456"}

            # Invoke the command with config file
            result = cli_runner.invoke(run_command, ["test_job", "--config", str(mock_config_file)])

            # Check the result
            assert result.exit_code == 0
            assert "success" in result.output

            # Verify mock was called with correct args
            mock_run_job.assert_called_once()
            args, kwargs = mock_run_job.call_args
            assert args[0] == "test_job"
            assert kwargs["params"]["algorithm"] == "kmeans"
            assert kwargs["params"]["n_clusters"] == 5

    def test_run_command_with_error(self, cli_runner: CliRunner) -> None:
        """Test handling of errors during job execution."""
        with patch("clustering.cli.commands.run_job") as mock_run_job:
            # Set up the mock to raise an exception
            mock_run_job.side_effect = ValueError("Invalid job configuration")

            # Invoke the command
            result = cli_runner.invoke(run_command, ["invalid_job"])

            # Check the result - should contain error message regardless of exit code
            assert "Error running job" in result.output
            assert "Invalid job configuration" in result.output


class TestValidateCommand:
    """Tests for the 'validate' command."""

    def test_validate_config_file(self, cli_runner: CliRunner, mock_config_file: Path) -> None:
        """Test validation of a config file."""
        with patch("clustering.cli.commands.validate_config") as mock_validate:
            # Set up the mock
            mock_validate.return_value = {"valid": True, "message": "Config is valid"}

            # Invoke the command
            result = cli_runner.invoke(validate_command, ["--config", str(mock_config_file)])

            # Check the result
            assert result.exit_code == 0
            assert "valid" in result.output

            # Verify mock was called with correct file
            mock_validate.assert_called_once_with(str(mock_config_file))

    def test_validate_invalid_config(self, cli_runner: CliRunner) -> None:
        """Test validation of an invalid config."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as temp:
            # Create an invalid config (missing required fields)
            temp_path = Path(temp.name)
            json.dump({"job": {"kind": "test_job"}}, temp)  # Missing required params

        try:
            with patch("clustering.cli.commands.validate_config") as mock_validate:
                # Set up the mock to report invalid config
                mock_validate.return_value = {
                    "valid": False,
                    "message": "Missing required fields",
                    "errors": ["params is required"],
                }

                # Invoke the command
                result = cli_runner.invoke(validate_command, ["--config", str(temp_path)])

                # Check the result - should display errors
                assert "Missing required fields" in result.output
                assert "params is required" in result.output

        finally:
            if temp_path.exists():
                os.unlink(temp_path)

    def test_validate_data_file(self, cli_runner: CliRunner, sample_data_file: Path) -> None:
        """Test validation of a data file."""
        with patch("clustering.cli.commands.validate_data") as mock_validate:
            # Set up the mock
            mock_validate.return_value = {
                "valid": True,
                "message": "Data is valid",
                "rows": 3,
                "columns": ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"],
            }

            # Invoke the command
            result = cli_runner.invoke(
                validate_command, ["--data", str(sample_data_file), "--schema", "sales"]
            )

            # Check the result
            assert result.exit_code == 0
            assert "valid" in result.output
            assert "3 rows" in result.output

            # Verify mock was called with correct arguments
            mock_validate.assert_called_once()
            args, kwargs = mock_validate.call_args
            assert args[0] == str(sample_data_file)
            assert kwargs["schema"] == "sales"


class TestExportCommand:
    """Tests for the 'export' command."""

    def test_export_results(self, cli_runner: CliRunner) -> None:
        """Test exporting results to a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "results.csv")

            with patch("clustering.cli.commands.export_results") as mock_export:
                # Set up the mock
                mock_export.return_value = {
                    "success": True,
                    "message": "Exported 100 results",
                    "path": output_path,
                }

                # Invoke the command
                result = cli_runner.invoke(
                    export_command, ["test-job-123", "--output", output_path, "--format", "csv"]
                )

                # Check the result
                assert result.exit_code == 0
                assert "Exported 100 results" in result.output

                # Verify mock was called with correct args
                mock_export.assert_called_once()
                args, kwargs = mock_export.call_args
                assert args[0] == "test-job-123"
                assert kwargs["output_path"] == output_path
                assert kwargs["format"] == "csv"

    def test_export_with_filtering(self, cli_runner: CliRunner) -> None:
        """Test exporting results with filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "filtered_results.csv")

            with patch("clustering.cli.commands.export_results") as mock_export:
                # Set up the mock
                mock_export.return_value = {
                    "success": True,
                    "message": "Exported 50 filtered results",
                    "path": output_path,
                }

                # Invoke the command with filtering
                result = cli_runner.invoke(
                    export_command,
                    [
                        "test-job-123",
                        "--output",
                        output_path,
                        "--format",
                        "csv",
                        "--filter",
                        "cluster=1",
                    ],
                )

                # Check the result
                assert result.exit_code == 0
                assert "Exported 50 filtered results" in result.output

                # Verify mock was called with filter
                mock_export.assert_called_once()
                args, kwargs = mock_export.call_args
                assert kwargs["filter_expr"] == "cluster=1"


class TestStatusCommand:
    """Tests for the 'status' command."""

    def test_job_status(self, cli_runner: CliRunner) -> None:
        """Test getting the status of a job."""
        with patch("clustering.cli.commands.get_job_status") as mock_status:
            # Set up the mock
            mock_status.return_value = {
                "job_id": "test-job-123",
                "status": "COMPLETED",
                "start_time": "2023-01-01T12:00:00",
                "end_time": "2023-01-01T12:15:00",
                "duration": "15m 0s",
            }

            # Invoke the command
            result = cli_runner.invoke(status_command, ["test-job-123"])

            # Check the result
            assert result.exit_code == 0
            assert "COMPLETED" in result.output
            assert "test-job-123" in result.output

            # Verify mock was called with correct job id
            mock_status.assert_called_once_with("test-job-123")

    def test_status_with_json_output(self, cli_runner: CliRunner) -> None:
        """Test getting status with JSON output format."""
        with patch("clustering.cli.commands.get_job_status") as mock_status:
            # Set up the mock
            job_status = {
                "job_id": "test-job-123",
                "status": "RUNNING",
                "start_time": "2023-01-01T12:00:00",
                "steps": [
                    {"name": "load_data", "status": "COMPLETED"},
                    {"name": "process_data", "status": "RUNNING"},
                    {"name": "export_results", "status": "PENDING"},
                ],
            }
            mock_status.return_value = job_status

            # Invoke the command with JSON output
            result = cli_runner.invoke(status_command, ["test-job-123", "--output", "json"])

            # Should return valid JSON
            assert result.exit_code == 0

            # Mock should be called with the job ID
            mock_status.assert_called_once_with("test-job-123")


class TestListCommand:
    """Tests for the 'list' command."""

    def test_list_jobs(self, cli_runner: CliRunner) -> None:
        """Test listing jobs."""
        with patch("clustering.cli.commands.list_jobs") as mock_list:
            # Set up the mock
            mock_list.return_value = [
                {
                    "job_id": "job-001",
                    "type": "clustering",
                    "status": "COMPLETED",
                    "created_at": "2023-01-01T10:00:00",
                },
                {
                    "job_id": "job-002",
                    "type": "preprocessing",
                    "status": "RUNNING",
                    "created_at": "2023-01-01T11:30:00",
                },
            ]

            # Invoke the command
            result = cli_runner.invoke(list_command)

            # Check the result
            assert result.exit_code == 0
            assert "job-001" in result.output
            assert "job-002" in result.output
            assert "clustering" in result.output
            assert "preprocessing" in result.output

            # Verify mock was called with default params
            mock_list.assert_called_once()
            args, kwargs = mock_list.call_args
            assert kwargs["limit"] == 10

    def test_list_with_filtering(self, cli_runner: CliRunner) -> None:
        """Test listing jobs with filtering."""
        with patch("clustering.cli.commands.list_jobs") as mock_list:
            # Set up the mock
            mock_list.return_value = [
                {
                    "job_id": "job-003",
                    "type": "clustering",
                    "status": "COMPLETED",
                    "created_at": "2023-01-02T09:15:00",
                }
            ]

            # Invoke the command with filtering
            result = cli_runner.invoke(
                list_command, ["--type", "clustering", "--status", "COMPLETED"]
            )

            # Check the result
            assert result.exit_code == 0
            assert "job-003" in result.output
            assert "COMPLETED" in result.output

            # Verify mock was called with filters
            mock_list.assert_called_once()
            args, kwargs = mock_list.call_args
            assert kwargs["job_type"] == "clustering"
            assert kwargs["status"] == "COMPLETED"

    def test_list_with_limit(self, cli_runner: CliRunner) -> None:
        """Test listing jobs with a limit."""
        with patch("clustering.cli.commands.list_jobs") as mock_list:
            # Set up the mock to return many jobs
            mock_list.return_value = [
                {"job_id": f"job-{i}", "status": "COMPLETED"}
                for i in range(1, 6)  # Only return 5 jobs
            ]

            # Invoke the command with a limit
            result = cli_runner.invoke(list_command, ["--limit", "5"])

            # Check the result - should only show 5 jobs
            assert result.exit_code == 0
            assert "job-1" in result.output
            assert "job-5" in result.output
            assert "job-10" not in result.output

            # Verify mock was called with limit
            mock_list.assert_called_once()
            args, kwargs = mock_list.call_args
            assert kwargs["limit"] == 5
