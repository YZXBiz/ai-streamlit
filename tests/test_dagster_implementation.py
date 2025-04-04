"""Tests for the Dagster implementation."""

import sys
from pathlib import Path

import dagster as dg
import pytest

# Add package directory to path if not already installed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clustering.dagster import create_definitions  # noqa: E402
from clustering.dagster.assets.preprocessing.internal import merged_internal_data  # noqa: E402


@pytest.fixture
def dagster_defs() -> dg.Definitions:
    """Fixture for Dagster definitions.

    Returns:
        Dagster definitions
    """
    return create_definitions(env="dev")


def test_dagster_definitions_creation(dagster_defs: dg.Definitions) -> None:
    """Test that Dagster definitions can be created.

    Args:
        dagster_defs: Dagster definitions
    """
    # Check that the definitions were created
    assert dagster_defs is not None

    # Check that the assets were added
    assert len(dagster_defs.get_all_asset_nodes()) > 0

    # Check that the jobs were added
    assert len(dagster_defs.get_all_job_defs()) > 0

    # Check that the schedules were added
    assert len(dagster_defs.get_all_schedule_defs()) > 0


def test_asset_dependencies() -> None:
    """Test that asset dependencies are correctly defined."""
    # Check that merged_internal_data depends on internal_sales_data and internal_need_state_data
    merged_deps = merged_internal_data.key_dependencies()
    merged_deps_keys = [dep.path for dep in merged_deps]

    assert ["raw", "internal_sales_data"] in merged_deps_keys
    assert ["raw", "internal_need_state_data"] in merged_deps_keys


@pytest.mark.parametrize(
    "job_name",
    [
        "internal_preprocessing_job",
        "internal_clustering_job",
    ],
)
def test_job_definitions(dagster_defs: dg.Definitions, job_name: str) -> None:
    """Test that job definitions are correctly created.

    Args:
        dagster_defs: Dagster definitions
        job_name: Name of the job to test
    """
    # Get the job
    job = dagster_defs.get_job_def(job_name)
    assert job is not None


def test_internal_preprocessing_job_structure(dagster_defs: dg.Definitions) -> None:
    """Test that the internal preprocessing job has the correct structure.

    Args:
        dagster_defs: Dagster definitions
    """
    # Get the job
    job = dagster_defs.get_job_def("internal_preprocessing_job")
    assert job is not None

    # Check the job structure
    job_assets = job.asset_selection_data.assets

    # Check that the job includes all the required assets
    assert any(["internal_sales_data"] == asset.key.path for asset in job_assets)
    assert any(["internal_need_state_data"] == asset.key.path for asset in job_assets)
    assert any(["merged_internal_data"] == asset.key.path for asset in job_assets)
    assert any(["preprocessed_internal_sales"] == asset.key.path for asset in job_assets)
    assert any(["preprocessed_internal_sales_percent"] == asset.key.path for asset in job_assets)


def test_io_manager_registration(dagster_defs: dg.Definitions) -> None:
    """Test that the IO manager is registered.

    Args:
        dagster_defs: Dagster definitions
    """
    # Check that the IO manager is registered
    resources = dagster_defs.resources
    assert "io_manager" in resources


def test_config_resource_registration(dagster_defs: dg.Definitions) -> None:
    """Test that the configuration resource is registered.

    Args:
        dagster_defs: Dagster definitions
    """
    # Check that the configuration resource is registered
    resources = dagster_defs.resources
    assert "config" in resources
