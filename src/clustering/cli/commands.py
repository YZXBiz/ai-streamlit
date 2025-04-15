"""Command-line interface for the clustering pipeline.

This module provides a user-friendly CLI for interacting with the Dagster-based
clustering pipeline. It allows users to run jobs, manage sensors, and access the web UI.
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, cast

import click
import dagster as dg
from dagster._core.instance import DagsterInstance

s
from clustering.dagster import create_definitions
from clustering.infra import CONFIG

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


from clustering.dagster.sensors import (
    external_data_sensor,
    internal_clustering_complete_sensor,
    internal_data_sensor,
)


def load_env_file(env: str = "dev") -> bool:
    """Load environment variables from .env file.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Whether the .env file was loaded successfully
    """
    # First try environment-specific .env file
    root_dir = Path(__file__).parent.parent.parent.parent  # Get to project root
    env_file = root_dir / f".env.{env}"
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
    env: str = "dev",
    tags: dict[str, str] | None = None,
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
    try:
        definitions = create_definitions(env)
    except Exception as e:
        raise ValueError(f"Failed to create definitions for environment '{env}': {e}")

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


# Define available sensors for reference throughout the CLI
AVAILABLE_SENSORS = {
    "internal_data": internal_data_sensor,
    "external_data": external_data_sensor,
    "internal_clustering_complete": internal_clustering_complete_sensor,
}


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Clustering Pipeline CLI.

    A command-line interface for managing the clustering pipeline, including
    running jobs, managing sensors, and accessing the web UI.
    """
    pass


@main.group()
def sensor():
    """Manage clustering pipeline sensors.

    Sensors monitor for changes and trigger pipeline runs when needed.
    """
    pass


@sensor.command()
@click.argument("sensor_name", required=False)
def start(sensor_name: str | None):
    """Start a sensor by name or all sensors if no name is provided.

    SENSOR_NAME: Optional name of the sensor to start
    """
    if sensor_name:
        if sensor_name not in AVAILABLE_SENSORS:
            click.secho(
                f"Error: Sensor '{sensor_name}' not found. Available sensors: {', '.join(AVAILABLE_SENSORS.keys())}",
                fg="red",
            )
            sys.exit(1)

        # Start specific sensor
        click.secho(f"Starting sensor: {sensor_name}", fg="green")
        try:
            # Use the Dagster Python API instead of subprocess
            instance = DagsterInstance.get()
            sensor_def = AVAILABLE_SENSORS[sensor_name]
            instance.start_sensor(sensor_def)
            click.secho(f"Sensor '{sensor_name}' started successfully", fg="green")
        except Exception as e:
            click.secho(f"Failed to start sensor '{sensor_name}': {e}", fg="red")
            sys.exit(1)
    else:
        # Start all sensors
        click.secho("Starting all sensors...", fg="green")
        instance = DagsterInstance.get()
        for name, sensor_def in AVAILABLE_SENSORS.items():
            try:
                instance.start_sensor(sensor_def)
                click.secho(f"Sensor '{name}' started successfully", fg="green")
            except Exception as e:
                click.secho(f"Failed to start sensor '{name}': {e}", fg="red")


@sensor.command()
@click.argument("sensor_name", required=False)
def stop(sensor_name: str | None):
    """Stop a sensor by name or all sensors if no name is provided.

    SENSOR_NAME: Optional name of the sensor to stop
    """
    if sensor_name:
        if sensor_name not in AVAILABLE_SENSORS:
            click.secho(
                f"Error: Sensor '{sensor_name}' not found. Available sensors: {', '.join(AVAILABLE_SENSORS.keys())}",
                fg="red",
            )
            sys.exit(1)

        # Stop specific sensor
        click.secho(f"Stopping sensor: {sensor_name}", fg="yellow")
        try:
            instance = DagsterInstance.get()
            sensor_def = AVAILABLE_SENSORS[sensor_name]
            instance.stop_sensor(sensor_def)
            click.secho(f"Sensor '{sensor_name}' stopped successfully", fg="green")
        except Exception as e:
            click.secho(f"Failed to stop sensor '{sensor_name}': {e}", fg="red")
            sys.exit(1)
    else:
        # Stop all sensors
        click.secho("Stopping all sensors...", fg="yellow")
        instance = DagsterInstance.get()
        for name, sensor_def in AVAILABLE_SENSORS.items():
            try:
                instance.stop_sensor(sensor_def)
                click.secho(f"Sensor '{name}' stopped successfully", fg="green")
            except Exception as e:
                click.secho(f"Failed to stop sensor '{name}': {e}", fg="red")


@sensor.command(name="list")
def list_sensors():
    """List all sensors and their status."""
    click.secho("Listing sensors...", fg="blue")

    instance = DagsterInstance.get()
    for name, sensor_def in AVAILABLE_SENSORS.items():
        try:
            status = instance.get_sensor_state(sensor_def.get_sensor_origin_id())
            is_running = status and status.status == dg.SensorStatus.RUNNING
            status_str = "RUNNING" if is_running else "STOPPED"
            color = "green" if is_running else "red"
            click.secho(f"- {name}: {status_str}", fg=color)
        except Exception as e:
            click.secho(f"- {name}: ERROR - {e}", fg="red")


@sensor.command()
@click.argument("sensor_name")
def preview(sensor_name: str):
    """Preview a sensor execution without triggering runs.

    SENSOR_NAME: Name of the sensor to preview
    """
    if sensor_name not in AVAILABLE_SENSORS:
        click.secho(
            f"Error: Sensor '{sensor_name}' not found. Available sensors: {', '.join(AVAILABLE_SENSORS.keys())}",
            fg="red",
        )
        sys.exit(1)

    click.secho(f"Previewing sensor: {sensor_name}", fg="blue")

    # Use Dagster Python API instead of subprocess
    sensor_def = AVAILABLE_SENSORS[sensor_name]
    instance = DagsterInstance.get()

    try:
        context = dg.SensorEvaluationContext(
            instance_ref=instance.get_ref(),
            sensor_name=sensor_def.name,
            last_completion_time=None,
            last_run_key=None,
            cursor=None,
        )

        result = sensor_def.evaluate_tick(context)

        if result.run_requests:
            click.secho(f"\nSensor would create {len(result.run_requests)} run(s):", fg="green")
            for i, request in enumerate(result.run_requests):
                click.secho(f"\n--- Run {i + 1} ---", fg="blue")
                click.secho(f"Job: {request.job_name}", fg="blue")
                click.secho(f"Tags: {request.tags}", fg="blue")
                click.secho(f"Run config: {request.run_config}", fg="blue")
        else:
            click.secho("\nSensor would not create any runs at this time.", fg="yellow")

        if result.cursor:
            click.secho(f"\nSensor would update cursor to: {result.cursor}", fg="blue")
    except Exception as e:
        click.secho(f"Error evaluating sensor: {e}", fg="red")
        sys.exit(1)


@main.command()
@click.argument("job_name", type=str)
@click.option(
    "--env",
    type=click.Choice(["dev", "staging", "prod"]),
    default=CONFIG.env,
    help="Environment to use (dev, staging, prod)",
)
@click.option("--tags", type=str, multiple=True, help="Tags in format key=value to add to the run")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def run(job_name: str, env: str, tags: Tuple[str, ...], verbose: bool):
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
    type=click.Choice(["dev", "staging", "prod"]),
    default=CONFIG.env,
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
        definitions = create_definitions(CONFIG.env)

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
