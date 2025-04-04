"""Run a simple test to verify the pipeline configuration and IO manager.

This script runs a simple test pipeline that materializes just the internal_sales_data
asset to verify that the IO manager, readers, and asset configuration are working correctly.
"""

import logging
import os
import sys
from datetime import datetime

from dagster import DagsterInstance, materialize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add src directory to Python path if needed
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import from the clustering package
from clustering.dagster.assets.preprocessing.internal import internal_sales_data
from clustering.dagster.resources.data_io import data_reader
from clustering.dagster.resources.io_manager import clustering_io_manager


def run_simple_test():
    """Run a simple test pipeline to verify configuration and resources."""
    start_time = datetime.now()
    logger.info(f"Starting simple test at {start_time}")

    # Create a temporary output directory for this test
    output_dir = os.path.join("outputs", "test", start_time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    try:
        # Define resources for this test
        resources = {
            "io_manager": clustering_io_manager.configured({"base_dir": output_dir}),
            "input_sales_reader": data_reader.configured(
                {"kind": "ParquetReader", "config": {"path": "data/raw/sales.parquet"}}
            ),
            "config": None,  # Not needed for this simple test
        }

        # Create an ephemeral Dagster instance to avoid needing DAGSTER_HOME
        instance = DagsterInstance.ephemeral()

        # Materialize the internal_sales_data asset
        logger.info("Materializing internal_sales_data asset...")
        result = materialize(
            [internal_sales_data],
            resources=resources,
            instance=instance,
        )

        # Check for errors
        if result.success:
            logger.info("Asset materialization successful!")

            # Get the materialization events
            for event in result.get_asset_materialization_events():
                logger.info(f"Asset: {event.asset_key}")
                # Directly access the event data instead of metadata_entries
                logger.info(f"  Event type: {event.event_type_value}")

            # Verify output file exists
            output_path = os.path.join(output_dir, "internal_sales_data.parquet")
            if os.path.exists(output_path):
                logger.info(f"Output verified: {output_path}")
                file_size = os.path.getsize(output_path)
                logger.info(f"File size: {file_size} bytes")
            else:
                logger.warning(f"Output file not found at expected path: {output_path}")

            return True
        else:
            # Handle failures
            logger.error("Asset materialization failed!")
            for event in result.filter_events("FAILURE"):
                logger.error(f"Failure: {event}")
            return False

    except Exception as e:
        logger.exception(f"Error running simple test: {e}")
        return False
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Test completed in {duration:.2f} seconds")


if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1)
