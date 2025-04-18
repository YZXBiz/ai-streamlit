"""Dagster application entry point."""

import os
import tempfile
from datetime import datetime

import dagster as dg

from clustering.pipeline.definitions import defs


def get_dagster_home() -> str:
    """Get the DAGSTER_HOME directory based on environment.

    Returns:
        Path to Dagster home directory
    """
    dagster_home = os.environ.get("DAGSTER_HOME")
    if not dagster_home:
        # Create a temporary directory in the system's temp dir instead of source tree
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dagster_home = tempfile.mkdtemp(prefix=f"dagster_home_{timestamp}_")
        os.environ["DAGSTER_HOME"] = dagster_home

    # Ensure the directory exists
    os.makedirs(dagster_home, exist_ok=True)

    return dagster_home


def run_job(job_name: str = "full_pipeline_job", env: str = "dev") -> None:
    """Run the Dagster job.

    Args:
        job_name: Name of the job to run
        env: Environment to use (dev, staging, prod)
    """
    dagster_home = get_dagster_home()
    print(f"Using Dagster home: {dagster_home}")

    # Set environment variable
    os.environ["DAGSTER_ENV"] = env

    # Get the specified job
    job = defs.get_job_def(job_name)

    # Execute the job
    job.execute_in_process()


def run_app(host: str = "localhost", port: int = 3000, env: str = "dev") -> None:
    """Run the Dagster web UI.

    Args:
        host: Host to bind to
        port: Port to bind to
        env: Environment to use (dev, staging, prod)
    """
    # Set environment variable
    os.environ["DAGSTER_ENV"] = env

    # Get dagster home
    dagster_home = get_dagster_home()
    print(f"Using Dagster home: {dagster_home}")

    # Run webserver
    dg.webserver.run_webserver(
        host=host,
        port=port,
        workspace=dg.workspace.UnscopedDefinitionsWorkspace(defs),
    )
