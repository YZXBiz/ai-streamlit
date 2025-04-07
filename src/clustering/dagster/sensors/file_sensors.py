"""File-based sensors for triggering clustering pipelines."""

import glob
import os

import dagster as dg
from dagster import RunRequest, SensorEvaluationContext, SensorResult

from clustering.dagster.definitions import (
    define_external_clustering_job,
    define_internal_clustering_job,
)


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse an S3 path into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    path_without_scheme = s3_path[5:]
    parts = path_without_scheme.split("/", 1)

    if len(parts) < 2:
        return parts[0], ""

    return parts[0], parts[1]


@dg.sensor(
    job=define_internal_clustering_job(),
    minimum_interval_seconds=60,  # Check every minute
)
def new_internal_data_sensor(
    context: SensorEvaluationContext, s3_client=dg.ResourceParam(dg.InitResourceContext)
) -> SensorResult:
    """Sensor that triggers the internal clustering job when new data is available.

    Args:
        context: The sensor evaluation context
        s3_client: S3 client resource

    Returns:
        SensorResult containing run requests if new data is detected
    """
    # This assumes you have a config resource available
    config = context.resources.config.load("internal_clustering")
    input_path = config.get("io", {}).get("input_path", "")

    # Skip if no path configured
    if not input_path:
        return SensorResult(skip_reason="No input path configured")

    # Check if we have a cursor (the last processed file)
    cursor = context.cursor or ""

    # Check for new files
    if input_path.startswith("s3://"):
        bucket, prefix = _parse_s3_path(input_path)
        response = s3_client.list_objects(bucket=bucket, prefix=prefix)

        files = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            last_modified = obj["LastModified"].isoformat()
            files.append((f"s3://{bucket}/{key}", last_modified))
    else:
        # Local file system
        files = []
        for file_path in glob.glob(os.path.join(input_path, "*.csv")):
            last_modified = os.path.getmtime(file_path)
            files.append((file_path, str(last_modified)))

    # Sort by last modified time
    files.sort(key=lambda x: x[1])

    # Filter files that are newer than the cursor
    new_files = [f for f, ts in files if f > cursor]

    if not new_files:
        return SensorResult(skip_reason="No new files detected")

    # Take the most recent file that's newer than the cursor
    latest_file = new_files[-1]

    # Create a run request for the new file
    run_config = {"resources": {"io_manager": {"config": {"data_path": latest_file}}}}

    run_key = f"internal_clustering_{os.path.basename(latest_file)}"

    context.log.info(f"Detected new file: {latest_file}, triggering run with key {run_key}")

    # Update the cursor to the latest file
    return SensorResult(
        run_requests=[RunRequest(run_key=run_key, run_config=run_config)], cursor=latest_file
    )


@dg.sensor(
    job=define_external_clustering_job(),
    minimum_interval_seconds=300,  # Check every 5 minutes
)
def external_data_sensor(context: SensorEvaluationContext) -> SensorResult:
    """Sensor that triggers the external clustering job when external data is available.

    This sensor checks a designated directory or S3 path for new external data files.

    Args:
        context: The sensor evaluation context

    Returns:
        SensorResult containing run requests if new data is detected
    """
    # This would follow similar logic to the internal data sensor
    # but checking a different path for external data sources

    # For demonstration purposes, we'll use a simplified implementation
    config = context.resources.config.load("external_clustering")
    input_path = config.get("io", {}).get("input_path", "data/external/")

    # Skip if no path configured
    if not input_path:
        return SensorResult(skip_reason="No input path configured")

    cursor = context.cursor or ""

    # Just check for local files in this example
    if os.path.exists(input_path):
        files = []
        for file_path in glob.glob(os.path.join(input_path, "*.csv")):
            last_modified = os.path.getmtime(file_path)
            files.append((file_path, str(last_modified)))

        # Sort by last modified time
        files.sort(key=lambda x: x[1])

        # Filter files that are newer than the cursor
        new_files = [f for f, ts in files if f > cursor]

        if new_files:
            latest_file = new_files[-1]
            run_key = f"external_clustering_{os.path.basename(latest_file)}"

            context.log.info(f"Detected new external file: {latest_file}")

            run_config = {
                "resources": {"io_manager": {"config": {"external_data_path": latest_file}}}
            }

            # Update the cursor to the latest file
            return SensorResult(
                run_requests=[RunRequest(run_key=run_key, run_config=run_config)],
                cursor=latest_file,
            )

    return SensorResult(skip_reason="No new external files detected")
