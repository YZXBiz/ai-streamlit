"""Command-line interface for the clustering pipeline.

This module provides a user-friendly CLI for interacting with the Dagster-based
clustering pipeline. It allows users to run jobs, manage sensors, and access the web UI.
"""

import os
import sys
from pathlib import Path

import click
import dagster as dg
from dagster import ExecuteInProcessResult
from dagster._core.instance import DagsterInstance

from clustering.dagster import create_definitions
from clustering.infra import CONFIG, Environment

# Add dotenv import
try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(dotenv_path: str | Path | None = None) -> bool:
        """Fallback implementation if dotenv is not installed."""
        click.secho(
            "Warning: python-dotenv not installed. Environment variables from .env files will not be loaded.",
            fg="yellow",
        )
        return False


def load_env_file(env: str | Environment = Environment.DEV) -> bool:
    """Load environment variables from .env file.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Whether the .env file was loaded successfully
    """
    # Convert Environment enum to string if needed
    env_str = env.value if isinstance(env, Environment) else env
    
    # First try environment-specific .env file
    root_dir = Path(__file__).parent.parent.parent.parent  # Get to project root
    env_file = root_dir / f".env.{env_str}"
    if env_file.exists():
        click.secho(f"Loading environment variables from {env_file}", fg="green")
        return load_dotenv(dotenv_path=env_file)

    # Fall back to default .env file
    default_env_file = root_dir / ".env"
    if default_env_file.exists():
        click.secho(f"Loading environment variables from {default_env_file}", fg="green")
        return load_dotenv(dotenv_path=default_env_file)

    click.secho("No .env file found", fg="yellow")
    return False


def run_job(
    job_name: str,
    env: str | Environment = Environment.DEV,
    tags: dict[str, str] | None = None,
    raise_on_error: bool = False,
) -> ExecuteInProcessResult:
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
    # Convert Environment enum to string if needed
    env_str = env.value if isinstance(env, Environment) else env
    
    # Create definitions for the specified environment
    try:
        definitions = create_definitions(env_str)
    except Exception as e:
        raise ValueError(f"Failed to create definitions for environment '{env_str}': {e}") from e

    # Get the job
    job = definitions.get_job_def(job_name)
    if not job:
        available_jobs = [job_def.name for job_def in definitions.get_all_job_defs()]
        raise ValueError(f"Job '{job_name}' not found. Available jobs: {', '.join(available_jobs)}")

    # Set up tags
    all_tags = {"env": env_str}
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


def parse_tags(tag_strings: list[str]) -> dict[str, str]:
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
        else:
            click.secho(
                f"Warning: Ignoring invalid tag format: {tag}. Expected format: key=value",
                fg="yellow",
            )
    return tags


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Clustering Pipeline CLI.

    A command-line interface for managing the clustering pipeline, including
    running jobs, managing sensors, and accessing the web UI.
    """
    pass


@main.command()
@click.argument("job_name", type=str)
@click.option(
    "--env",
    type=click.Choice([e.value for e in Environment]),
    default=CONFIG.env.value,
    help="Environment to use (dev, staging, prod)",
)
@click.option("--tags", type=str, multiple=True, help="Tags in format key=value to add to the run")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def run(job_name: str, env: str, tags: tuple[str, ...], verbose: bool):
    """Run a Dagster job.

    JOB_NAME: Name of the job to run. Options include:
    - internal_preprocessing_job
    - internal_clustering_job
    - external_preprocessing_job
    - external_clustering_job
    - merging_job
    - full_pipeline_job
    """
    # Set environment variable
    os.environ["DAGSTER_ENV"] = env
    # Load environment variables from .env file
    load_env_file(env)

    # Parse tags
    tag_dict = parse_tags(list(tags)) if tags else {}

    # Run the job
    click.secho(f"Starting Dagster job '{job_name}' in {env.upper()} environment", fg="blue")
    if verbose:
        click.secho(f"Tags: {tag_dict}", fg="blue")

    try:
        result = run_job(job_name, env, tag_dict)

        # Check result
        if result.success:
            click.secho(f"✅ Job '{job_name}' completed successfully", fg="green")

            if verbose:
                click.secho("\nStep details:", fg="blue")
                for step_event in result.all_node_events:
                    if step_event.is_successful_output:
                        click.secho(f"  - {step_event.step_key}: SUCCESS", fg="green")

            sys.exit(0)
        else:
            click.secho(f"❌ Job '{job_name}' failed:", fg="red")
            for step_failure in result.all_node_events:
                if step_failure.is_failure:
                    click.secho(
                        f"  - Step '{step_failure.step_key}' failed: "
                        f"{step_failure.event_specific_data.error.message}",
                        fg="red",
                    )
            sys.exit(1)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error running job '{job_name}': {e}", fg="red")
        sys.exit(1)


@main.command()
@click.option("--host", type=str, default="localhost", help="Host to bind to")
@click.option("--port", type=int, default=3000, help="Port to bind to")
@click.option(
    "--env",
    type=click.Choice([e.value for e in Environment]),
    default=CONFIG.env.value,
    help="Environment to use (dev, staging, prod)",
)
def ui(host: str, port: int, env: str):
    """Launch the Dagster web UI.

    This provides a web interface for monitoring and managing the clustering pipeline.
    """
    # Set environment variable
    os.environ["DAGSTER_ENV"] = env
    # Load environment variables from .env file
    load_env_file(env)

    # Import here to avoid circular imports
    try:
        from clustering.dagster.app import run_app

        # Run the UI
        click.secho(
            f"Starting Dagster web UI in {env.upper()} environment on {host}:{port}", fg="blue"
        )
        click.secho(f"Access the UI at http://{host}:{port}", fg="green")
        run_app(host=host, port=port, env=env)
    except ImportError as e:
        click.secho(f"Error importing app module: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error launching UI: {e}", fg="red")
        sys.exit(1)


@main.command()
@click.option("--host", type=str, default="localhost", help="Host to bind to")
@click.option("--port", type=int, default=3000, help="Port to bind to")
def minimal(host: str, port: int):
    """Run minimal example with SQL engine.

    This launches a simplified version of the pipeline for demonstration purposes.
    """
    click.secho(f"Starting minimal Dagster example on {host}:{port}", fg="blue")
    click.secho(f"Access the UI at http://{host}:{port}", fg="green")

    try:
        dg.webserver.run_webserver(
            host=host,
            port=port,
            workspace=dg.workspace.LoadableTargetWorkspace(
                loadable_target=dg.workspace.LoadableTarget(
                    attribute="defs", python_module="clustering.dagster.minimal_example"
                )
            ),
        )
    except Exception as e:
        click.secho(f"Error running minimal example: {e}", fg="red")
        sys.exit(1)


@main.command()
def list_jobs():
    """List all available jobs in the Dagster repository."""
    click.secho("Listing available jobs:", fg="blue")

    try:
        # Create definitions for the current environment
        definitions = create_definitions(CONFIG.env.value)

        # Get all job definitions
        jobs = [job_def.name for job_def in definitions.get_all_job_defs()]

        if jobs:
            for job in sorted(jobs):
                click.secho(f"- {job}", fg="green")
        else:
            click.secho("No jobs found", fg="yellow")
    except Exception as e:
        click.secho(f"Error listing jobs: {e}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
