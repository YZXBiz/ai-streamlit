"""Command-line interface for the clustering pipeline."""

import argparse
import os
import sys
from typing import Dict, List, Optional

import dagster as dg
from dagster._core.instance import DagsterInstance

from clustering.dagster import create_definitions
from clustering.infra import CONFIG


def run_job(
    job_name: str,
    env: str = "dev",
    tags: Optional[Dict[str, str]] = None,
    raise_on_error: bool = False,
) -> dg.ExecuteJobResult:
    """Run a Dagster job.

    Args:
        job_name: Name of the job to run
        env: Environment to use (dev, staging, prod)
        tags: Tags to add to the run
        raise_on_error: Whether to raise on error

    Returns:
        Result of the job execution

    Raises:
        ValueError: If the job is not found
    """
    # Create definitions for the specified environment
    definitions = create_definitions(env)

    # Get the job
    job = definitions.get_job_def(job_name)
    if not job:
        available_jobs = [job_def.name for job_def in definitions.get_all_job_defs()]
        raise ValueError(f"Job '{job_name}' not found. Available jobs: {', '.join(available_jobs)}")

    # Set up tags
    all_tags = {"env": env}
    if tags:
        all_tags.update(tags)

    # Get the Dagster instance
    instance = DagsterInstance.get()

    # Execute the job
    result = job.execute_in_process(
        instance=instance,
        tags=all_tags,
        raise_on_error=raise_on_error,
    )

    return result


def parse_tags(tag_strings: List[str]) -> Dict[str, str]:
    """Parse tags from command line arguments.

    Args:
        tag_strings: List of strings in format key=value

    Returns:
        Dictionary of tags
    """
    tags = {}
    for tag in tag_strings:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key] = value
    return tags


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Run Dagster jobs for the clustering pipeline")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run job command
    run_parser = subparsers.add_parser("run", help="Run a Dagster job")
    run_parser.add_argument(
        "job_name",
        type=str,
        help=(
            "Job name to run. Options: internal_preprocessing_job, internal_clustering_job, "
            "external_preprocessing_job, external_clustering_job, merging_job, full_pipeline_job"
        ),
    )
    run_parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default=CONFIG.get_env(),
        help="Environment to use (dev, staging, prod)",
    )
    run_parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags in format key=value to add to the run",
    )

    # Web UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Dagster web UI")
    ui_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to bind to",
    )
    ui_parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default=CONFIG.get_env(),
        help="Environment to use (dev, staging, prod)",
    )

    # Add minimal example command
    minimal_parser = subparsers.add_parser("minimal", help="Run minimal example with SQL engine")
    minimal_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to",
    )
    minimal_parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to bind to",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set environment variable for environment
    if hasattr(args, "env"):
        os.environ["DAGSTER_ENV"] = args.env

    # Execute the requested command
    if args.command == "run":
        # Parse tags
        tags = parse_tags(args.tags) if args.tags else {}

        # Run the job
        print(f"Starting Dagster job '{args.job_name}' in {args.env.upper()} environment")
        result = run_job(args.job_name, args.env, tags)

        # Check result
        if result.success:
            print(f"✅ Job '{args.job_name}' completed successfully")
            sys.exit(0)
        else:
            print(f"❌ Job '{args.job_name}' failed:")
            for step_failure in result.all_node_events:
                if step_failure.is_failure:
                    print(
                        f"  - Step '{step_failure.step_key}' failed: "
                        f"{step_failure.event_specific_data.error.message}"
                    )
            sys.exit(1)

    elif args.command == "ui":
        # Import here to avoid circular imports
        from clustering.dagster.app import run_app

        # Run the UI
        print(f"Starting Dagster web UI in {args.env.upper()} environment on {args.host}:{args.port}")
        run_app(host=args.host, port=args.port, env=args.env)

    elif args.command == "minimal":
        # Import minimal example

        # Run UI with minimal example
        print(f"Starting minimal Dagster example on {args.host}:{args.port}")
        dg.webserver.run_webserver(
            host=args.host,
            port=args.port,
            workspace=dg.workspace.LoadableTargetWorkspace(
                loadable_target=dg.workspace.LoadableTarget(
                    attribute="defs", python_module="clustering.dagster.minimal_example"
                )
            ),
        )

    else:
        # No command specified, show help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
