#!/usr/bin/env python
"""
Script to run the Dagster clustering pipeline.

This script provides a command-line interface to run the Dagster clustering pipeline,
either the full pipeline or specific jobs.
"""

import argparse
import sys
from datetime import datetime

from dagster import DagsterInstance

# Import the Dagster definitions
sys.path.append(".")
from src.clustering.dagster.definitions import create_definitions


def run_job(job_name: str, env: str = "dev", tags: dict | None = None) -> str:
    """Run a Dagster job.

    Args:
        job_name: Name of the job to run
        env: Environment to run in (dev, staging, prod)
        tags: Additional tags to add to the run

    Returns:
        Run ID
    """
    if tags is None:
        tags = {}

    # Add default tags
    default_tags = {
        "env": env,
        "run_by": "script",
        "timestamp": datetime.now().isoformat(),
    }
    run_tags = {**default_tags, **tags}

    # Create definitions for the specified environment
    defs = create_definitions(env=env)

    # Get the job
    job = defs.get_job_def(job_name)
    if not job:
        available_jobs = [j.name for j in defs.get_all_jobs()]
        raise ValueError(f"Job '{job_name}' not found. Available jobs: {available_jobs}")

    # Get the instance
    instance = DagsterInstance.get()

    # Create and launch the run
    run_config = {}
    run_id = (
        defs.get_job_def(job_name)
        .execute_in_process(
            run_config=run_config,
            instance=instance,
            raise_on_error=True,
            tags=run_tags,
        )
        .run_id
    )

    print(f"Started job '{job_name}' with run ID: {run_id}")
    return run_id


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Dagster clustering pipeline")
    parser.add_argument(
        "--job",
        type=str,
        choices=[
            "internal_preprocessing_job",
            "internal_clustering_job",
            "external_preprocessing_job",
            "external_clustering_job",
            "merging_job",
            "full_pipeline_job",
        ],
        default="full_pipeline_job",
        help="The job to run (default: full_pipeline_job)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["dev", "staging", "prod"],
        default="dev",
        help="The environment to run in (default: dev)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Additional tags in format key=value (can be specified multiple times)",
    )

    args = parser.parse_args()

    # Parse tags
    tags = {}
    if args.tags:
        for tag in args.tags:
            try:
                key, value = tag.split("=", 1)
                tags[key] = value
            except ValueError:
                print(f"Warning: Ignoring malformed tag '{tag}'. Format should be key=value.")

    try:
        run_id = run_job(args.job, args.env, tags)
        print(f"Pipeline job '{args.job}' completed successfully with run ID: {run_id}")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
