"""Main entry point for the clustering pipeline."""

import os
import sys

from dagster import DagsterInstance

from clustering.dagster.definitions import create_definitions

if __name__ == "__main__":
    # Get environment from environment variable or command line argument
    env = os.environ.get("DAGSTER_ENV", "dev")

    if len(sys.argv) > 1:
        # Allow overriding environment from command line
        if sys.argv[1] in ["dev", "staging", "prod"]:
            env = sys.argv[1]

    print(f"Starting Dagster pipeline in {env.upper()} environment")

    # Create definitions with specified environment
    definitions = create_definitions(env=env)

    # Get the Dagster instance
    instance = DagsterInstance.get()

    # Default to the full pipeline job if no specific job is provided
    job_name = "full_pipeline_job"

    if len(sys.argv) > 2:
        # Allow specifying a specific job to run
        job_name = sys.argv[2]

    # Get the job
    job = definitions.get_job_def(job_name)

    # Execute the job
    result = job.execute_in_process(
        instance=instance,
        raise_on_error=False,
    )

    # Print the result
    if result.success:
        print(f"Job {job_name} completed successfully")
        sys.exit(0)
    else:
        print(f"Job {job_name} failed: {result.failure_data}")
        sys.exit(1)
