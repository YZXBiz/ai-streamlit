#!/usr/bin/env python
"""Command-line script for running Dagster jobs with environment selection."""

import argparse
import sys

import dagster as dg
from dagster._core.instance import DagsterInstance

from clustering.dagster import create_definitions


def main():
    """Run Dagster jobs from the command line."""
    parser = argparse.ArgumentParser(description="Run Dagster jobs for the clustering pipeline")
    parser.add_argument(
        "job_name",
        type=str,
        help=(
            "Job name to run. Options: internal_preprocessing_job, internal_clustering_job, "
            "external_preprocessing_job, external_clustering_job, merging_job, full_pipeline_job"
        ),
    )
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment to use (dev, staging, prod)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags in format key=value to add to the run",
    )

    args = parser.parse_args()

    # Create definitions for the specified environment
    definitions = create_definitions(args.env)

    # Get the job
    job = definitions.get_job_def(args.job_name)
    if not job:
        print(f"Error: Job '{args.job_name}' not found. Available jobs:")
        for job_def in definitions.get_all_job_defs():
            print(f"  - {job_def.name}")
        sys.exit(1)

    # Parse tags
    tags = {"env": args.env}
    if args.tags:
        for tag in args.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tags[key] = value

    # Run the job
    instance = DagsterInstance.get()
    result = dg.execute_job(
        job,
        instance=instance,
        tags=tags,
    )

    # Check result
    if result.success:
        print(f"✅ Job '{args.job_name}' completed successfully")
        sys.exit(0)
    else:
        print(f"❌ Job '{args.job_name}' failed:")
        for step_failure in result.all_node_events:
            if step_failure.is_failure:
                print(f"  - Step '{step_failure.step_key}' failed: {step_failure.event_specific_data.error.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
