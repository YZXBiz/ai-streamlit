"""Dagster application entry point."""

import os

from dagster import DagsterInstance

from clustering.dagster.definitions import clustering_job


def get_dagster_home() -> str:
    """Get the DAGSTER_HOME directory based on environment.

    Returns:
        Path to Dagster home directory
    """
    dagster_home = os.environ.get("DAGSTER_HOME")
    if not dagster_home:
        # Create a temporary directory in the system's temp dir instead of source tree
        import tempfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dagster_home = tempfile.mkdtemp(prefix=f"dagster_home_{timestamp}_")
        os.environ["DAGSTER_HOME"] = dagster_home

    # Ensure the directory exists
    os.makedirs(dagster_home, exist_ok=True)

    return dagster_home


def run_job():
    """Run the Dagster job."""
    dagster_home = get_dagster_home()
    print(f"Using Dagster home: {dagster_home}")

    instance = DagsterInstance.get()
    clustering_job.execute_in_process(instance_ref=instance.get_ref())
