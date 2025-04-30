"""
Main application package.
"""

import logging
import os
from datetime import datetime


# Initialize logging
def setup_logging():
    """Set up logging for the application."""
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a log file with timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"app-{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info("Logging initialized")


# Set up logging when app is imported
setup_logging()

__version__ = "0.1.0"

# App package initialization
# App package initialization
