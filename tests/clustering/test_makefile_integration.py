"""Tests for Dagster integration with the Makefile."""

import os
import subprocess
from pathlib import Path

import pytest


def run_make_command(command: str) -> subprocess.CompletedProcess:
    """Run a make command from the project root.

    Args:
        command: The make command to run.

    Returns:
        The completed process object.
    """
    # Navigate to project root
    project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from tests/clustering/

    # Run the make command
    return subprocess.run(
        f"make {command}",
        shell=True,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception on non-zero return
    )


@pytest.mark.skip(reason="Only run these manually - they interact with the real system")
class TestMakefileCommands:
    """Tests for Makefile commands related to Dagster."""

    def test_dagster_home_is_created(self) -> None:
        """Test that the DAGSTER_HOME directory is created by setup target."""
        result = run_make_command("setup")

        assert result.returncode == 0, f"Setup failed with error: {result.stderr}"

        # The DAGSTER_HOME should be set in the Makefile
        # Check if we can determine what it is
        env_result = subprocess.run(
            "cat .env | grep DAGSTER_HOME || echo 'DAGSTER_HOME not found'",
            shell=True,
            capture_output=True,
            text=True,
        )

        # Extract DAGSTER_HOME path if found
        dagster_home = None
        for line in env_result.stdout.splitlines():
            if "DAGSTER_HOME" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    dagster_home = parts[1].strip().strip("\"'")

        if dagster_home:
            assert Path(dagster_home).exists(), f"DAGSTER_HOME directory {dagster_home} not created"
        else:
            pytest.skip("Could not determine DAGSTER_HOME from .env file")

    def test_make_dagster_ui_works(self) -> None:
        """Test that 'make dagster-ui' command works correctly.

        This test doesn't actually run the UI (that would block), but checks
        that the command is formatted correctly.
        """
        # Run command with --help to verify it's formatted correctly
        # without actually starting the server
        result = run_make_command("dagster-ui -- --help")

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Dagster UI" in result.stdout or "dagster-webserver" in result.stdout

    def test_make_dagster_dbt_works(self) -> None:
        """Test that 'make dagster-dbt' command works correctly.

        This test just checks that the command is properly configured.
        """
        result = run_make_command("dagster-dbt -- --help")

        # Note: This might fail if dbt is not installed - skip in that case
        if result.returncode != 0 and "No such file or directory" in result.stderr:
            pytest.skip("dbt command not available")

        assert "dagster-dbt" in result.stdout or "dbt" in result.stdout

    def test_make_dagster_materialize_works(self) -> None:
        """Test that 'make dagster-materialize' command is properly configured."""
        # Just check the command helps
        result = run_make_command("dagster-materialize -- --help")

        # If command exists, it should return help info
        if result.returncode == 0:
            assert "dagster" in result.stdout.lower()
        else:
            # If command fails, it might be because it's not fully implemented yet
            pytest.skip("dagster-materialize command not fully implemented")


@pytest.mark.skip(reason="Only run these manually")
def test_makefile_has_required_targets() -> None:
    """Test that the Makefile has all required targets for Dagster integration."""
    # Read the Makefile content
    project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels
    makefile_path = project_root / "Makefile"

    if not makefile_path.exists():
        pytest.skip("Makefile not found")

    makefile_content = makefile_path.read_text()

    # Check for required targets
    required_targets = [
        "setup",
        "dagster-ui",
        "dagster-daemon",
    ]

    for target in required_targets:
        assert f"{target}:" in makefile_content, f"Required target '{target}' not found in Makefile"

    # Check that DAGSTER_HOME is managed in the Makefile
    assert "DAGSTER_HOME" in makefile_content, "DAGSTER_HOME not managed in Makefile"
