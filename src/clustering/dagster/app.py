"""Entry point for running the Dagster webserver."""

from dagster import DagsterInstance
from dagster._cli.ui import ui_cli

from clustering.dagster import create_definitions


def run_app(host: str = "127.0.0.1", port: int = 3000, env: str = "dev"):
    """Run the Dagster webserver.

    Args:
        host: Host address to bind the webserver to
        port: Port to run the Dagster webserver on
        env: Environment to use (dev, staging, prod)
    """
    print(f"Starting Dagster webserver with {env} environment on {host}:{port}")

    # Create definitions for the specified environment
    defs = create_definitions(env)

    # Run the Dagster webserver
    instance = DagsterInstance.get()
    ui_cli.ui(
        defs,
        instance,
        port=port,
        host=host,
    )
