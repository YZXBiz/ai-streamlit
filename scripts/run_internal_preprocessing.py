#!/usr/bin/env python
"""
A script to run the entire internal preprocessing job.
"""

import os
import sys

from dagster import DagsterInstance

# Import the Dagster definitions
sys.path.append(".")
from src.clustering.dagster.definitions import create_definitions


def run_internal_preprocessing():
    """Run the internal preprocessing job with all assets."""
    # Create definitions with dev environment
    defs = create_definitions(env="dev")

    # Make sure storage directory exists
    os.makedirs("storage", exist_ok=True)

    # Get the internal preprocessing job
    internal_preprocessing_job = next(
        job for job in defs.get_all_job_defs() if job.name == "internal_preprocessing_job"
    )

    # Get an ephemeral instance
    instance = DagsterInstance.ephemeral()

    # Run the job
    print(f"Running job: {internal_preprocessing_job.name}")
    result = internal_preprocessing_job.execute_in_process(
        instance=instance,
        raise_on_error=False,
    )

    if result.success:
        print(f"Job {internal_preprocessing_job.name} completed successfully!")
        return result
    else:
        print(f"Job {internal_preprocessing_job.name} failed!")
        # Print the failure information from the result
        for event in result.all_node_events:
            if event.is_step_failure:
                error_msg = event.message
                if hasattr(event, "event_specific_data"):
                    if hasattr(event.event_specific_data, "error"):
                        error_msg = f"{error_msg}: {event.event_specific_data.error}"
                print(f"Error: {error_msg}")
        return result


if __name__ == "__main__":
    run_internal_preprocessing()
