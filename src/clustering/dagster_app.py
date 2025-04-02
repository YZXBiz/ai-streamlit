"""Entry point for running the Dagster webserver."""

import argparse
import sys
from pathlib import Path

# Add package directory to path if not already installed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dagster import DagsterInstance
from dagster._cli.ui import ui_cli

from clustering.dagster import create_definitions


def main():
    """Run the Dagster webserver."""
    parser = argparse.ArgumentParser(description="Run the Dagster webserver for clustering project")
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment to use (dev, staging, prod)",
    )
    parser.add_argument("--port", type=int, default=3000, help="Port to run the Dagster webserver on")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind the webserver to")

    args = parser.parse_args()

    print(f"Starting Dagster webserver with {args.env} environment")

    # Create definitions for the specified environment
    defs = create_definitions(args.env)

    # Run the Dagster webserver
    instance = DagsterInstance.get()
    ui_cli.ui(
        defs,
        instance,
        port=args.port,
        host=args.host,
    )


if __name__ == "__main__":
    main()
