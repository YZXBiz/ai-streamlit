"""Entry point for running the Dagster webserver."""

import os
import subprocess


def run_app(
    host: str = "127.0.0.1",
    port: int = 3000,
    env: str = "dev",
):
    """Run the Dagster webserver."""
    print(f"Starting Dagster webserver with {env} environment on {host}:{port}")

    os.environ["DAGSTER_ENV"] = env

    subprocess.run(["dagster-webserver", "-h", host, "-p", str(port)])
