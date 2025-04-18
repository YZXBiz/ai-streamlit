"""Command-line entry point for the clustering dashboard."""

import os
import sys

import streamlit.web.cli as stcli


def main() -> None:
    """Run the dashboard as a module.
    
    This function sets up the Streamlit CLI arguments and launches the dashboard.
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the path to app.py
    app_path = os.path.join(current_dir, "app.py")

    # Run the Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.serverAddress=0.0.0.0",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main() 