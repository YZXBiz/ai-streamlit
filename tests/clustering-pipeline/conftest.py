"""Test fixtures for the pipeline package."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from dagster import (
    Definitions,
)
from dagster._core.test_utils import InProcessExecutionResult


@pytest.fixture
def test_data_dir() -> Path:
    """Create a temporary directory for test data.

    Returns:
        Path: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create standard subdirectories
        (temp_path / "input").mkdir()
        (temp_path / "output").mkdir()

        # Yield the path to the temporary directory
        yield temp_path


@pytest.fixture
def mock_data() -> pd.DataFrame:
    """Create mock data for tests.

    Returns:
        pd.DataFrame: A DataFrame with mock data.
    """
    # Create a simple mock dataset
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "C", "B"],
        }
    )


@pytest.fixture
def mock_pipeline_context() -> dict[str, Any]:
    """Create a mock context for pipeline execution.

    Returns:
        dict: A dictionary with context information.
    """
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()

    # Create a context dictionary
    context = {
        "temp_dir": temp_dir,
        "env": "test",
        "params": {"algorithm": "kmeans", "n_clusters": 3, "random_state": 42},
    }

    return context


@pytest.fixture
def run_dagster_job() -> Callable[..., InProcessExecutionResult]:
    """Create a fixture for running a Dagster job.

    Returns:
        Callable: A function to run a Dagster job.
    """

    def _run_job(defs: Definitions, job_name: str, **kwargs: Any) -> InProcessExecutionResult:
        """Run a Dagster job.

        Args:
            defs: Dagster definitions containing the job.
            job_name: Name of the job to run.
            **kwargs: Additional keyword arguments to pass to execute_job.

        Returns:
            InProcessExecutionResult: The result of executing the job.
        """
        job = defs.get_job_def(job_name)
        return job.execute_in_process(**kwargs)

    return _run_job
