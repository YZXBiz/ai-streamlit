"""Tests for the Dagster application entry point."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from clustering.pipeline.app import get_dagster_home, run_app, run_job


@pytest.fixture
def mock_env_vars():
    """Fixture to manage environment variables temporarily for tests."""
    # Save original environment variable values
    original_dagster_home = os.environ.get("DAGSTER_HOME")
    original_dagster_env = os.environ.get("DAGSTER_ENV")

    # Clean up after test
    yield

    # Restore original environment variables
    if original_dagster_home:
        os.environ["DAGSTER_HOME"] = original_dagster_home
    else:
        os.environ.pop("DAGSTER_HOME", None)

    if original_dagster_env:
        os.environ["DAGSTER_ENV"] = original_dagster_env
    else:
        os.environ.pop("DAGSTER_ENV", None)


class TestGetDagsterHome:
    """Tests for get_dagster_home function."""

    def test_get_dagster_home_from_env(self, mock_env_vars):
        """Test that it returns DAGSTER_HOME when already set."""
        # Set environment variable
        test_dir = "/tmp/test_dagster_home"
        os.environ["DAGSTER_HOME"] = test_dir

        # Call function
        result = get_dagster_home()

        # Verify result
        assert result == test_dir

    @patch("tempfile.mkdtemp")
    def test_get_dagster_home_creates_temp_dir(self, mock_mkdtemp, mock_env_vars):
        """Test that it creates a temp directory when DAGSTER_HOME is not set."""
        # Remove DAGSTER_HOME if it exists
        if "DAGSTER_HOME" in os.environ:
            del os.environ["DAGSTER_HOME"]

        # Set up mock
        mock_temp_dir = "/tmp/mock_dagster_home_12345"
        mock_mkdtemp.return_value = mock_temp_dir

        # Call function
        result = get_dagster_home()

        # Verify result
        assert result == mock_temp_dir
        assert os.environ["DAGSTER_HOME"] == mock_temp_dir
        mock_mkdtemp.assert_called_once()


class TestRunJob:
    """Tests for run_job function."""

    @patch("clustering.pipeline.app.get_dagster_home")
    @patch("clustering.pipeline.app.defs")
    def test_run_job_with_defaults(self, mock_defs, mock_get_home, mock_env_vars):
        """Test running a job with default parameters."""
        # Set up mocks
        mock_job = MagicMock()
        mock_defs.get_job_def.return_value = mock_job
        mock_get_home.return_value = "/tmp/dagster_home"

        # Call function
        run_job()

        # Verify function calls and environment variable
        mock_get_home.assert_called_once()
        mock_defs.get_job_def.assert_called_once_with("full_pipeline_job")
        mock_job.execute_in_process.assert_called_once()
        assert os.environ["DAGSTER_ENV"] == "dev"

    @patch("clustering.pipeline.app.get_dagster_home")
    @patch("clustering.pipeline.app.defs")
    def test_run_job_with_custom_params(self, mock_defs, mock_get_home, mock_env_vars):
        """Test running a job with custom parameters."""
        # Set up mocks
        mock_job = MagicMock()
        mock_defs.get_job_def.return_value = mock_job
        mock_get_home.return_value = "/tmp/dagster_home"

        # Call function with custom parameters
        run_job(job_name="test_job", env="prod")

        # Verify function calls and environment variable
        mock_get_home.assert_called_once()
        mock_defs.get_job_def.assert_called_once_with("test_job")
        mock_job.execute_in_process.assert_called_once()
        assert os.environ["DAGSTER_ENV"] == "prod"


class TestRunApp:
    """Tests for run_app function."""

    @patch("clustering.pipeline.app.get_dagster_home")
    def test_run_app_with_defaults(self, mock_get_home, mock_env_vars):
        """Test running app with default parameters."""
        # Set up mocks
        mock_get_home.return_value = "/tmp/dagster_home"

        # Mock the run_webserver function without actually calling it
        with patch("clustering.pipeline.app.dg") as mock_dg:
            # Setup the mock
            mock_workspace = MagicMock()
            mock_dg.workspace.UnscopedDefinitionsWorkspace.return_value = mock_workspace

            # Call function
            run_app()

            # Verify function calls
            mock_get_home.assert_called_once()
            mock_dg.webserver.run_webserver.assert_called_once_with(
                host="localhost", port=3000, workspace=mock_workspace
            )

        # Verify environment variable was set
        assert os.environ["DAGSTER_ENV"] == "dev"

    @patch("clustering.pipeline.app.get_dagster_home")
    def test_run_app_with_custom_params(self, mock_get_home, mock_env_vars):
        """Test running app with custom parameters."""
        # Set up mocks
        mock_get_home.return_value = "/tmp/dagster_home"

        # Mock the run_webserver function without actually calling it
        with patch("clustering.pipeline.app.dg") as mock_dg:
            # Setup the mock
            mock_workspace = MagicMock()
            mock_dg.workspace.UnscopedDefinitionsWorkspace.return_value = mock_workspace

            # Call function with custom parameters
            run_app(host="0.0.0.0", port=8080, env="prod")

            # Verify function calls
            mock_get_home.assert_called_once()
            mock_dg.webserver.run_webserver.assert_called_once_with(
                host="0.0.0.0", port=8080, workspace=mock_workspace
            )

        # Verify environment variable was set
        assert os.environ["DAGSTER_ENV"] == "prod"
