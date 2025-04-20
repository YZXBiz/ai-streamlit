"""Command-line entry point for the clustering dashboard."""

import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli


def main() -> None:
    """Run the dashboard as a module.

    This function sets up the Streamlit CLI arguments and launches the dashboard.
    It ensures the pages directory is properly recognized by Streamlit.
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the path to app.py
    app_path = os.path.join(current_dir, "app.py")

    # Make sure the pages directory is in a place Streamlit can find it
    os.path.join(current_dir, "pages")

    # Run the Streamlit app with the appropriate arguments
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=200",
        "--browser.gatherUsageStats=false",
    ]

    # Set environment variable to help with path resolution if needed
    os.environ["PYTHONPATH"] = (
        f"{os.environ.get('PYTHONPATH', '')}:{Path(current_dir).parent.parent}"
    )

    # Execute Streamlit CLI
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
