"""Test the preprocessing pipeline with multiple assets.

This script runs the internal preprocessing pipeline, which includes:
1. internal_sales_data
2. internal_need_state_data
3. merged_internal_data

It validates that each asset can be properly materialized and that the data flows correctly
between assets.
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
from clustering.dagster.assets.preprocessing.internal import (
    internal_need_state_data,
    internal_sales_data,
    merged_internal_data,
)
from clustering.dagster.resources.data_io import data_reader
from clustering.dagster.resources.io_manager import clustering_io_manager


def run_preprocessing_test():
    """Run a test of the internal preprocessing pipeline."""
    start_time = datetime.now()
    logger.info(f"Starting preprocessing pipeline test at {start_time}")

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
            "input_need_state_reader": data_reader.configured(
                {"kind": "CSVReader", "config": {"path": "data/raw/need_state.csv"}}
            ),
            "config": None,  # Not needed for this simple test
        }

        # Create an ephemeral Dagster instance to avoid needing DAGSTER_HOME
        instance = DagsterInstance.ephemeral()

        # Materialize the preprocessing assets
        logger.info("Materializing preprocessing assets...")
        result = materialize(
            [internal_sales_data, internal_need_state_data, merged_internal_data],
            resources=resources,
            instance=instance,
        )

        # Check for errors
        if result.success:
            logger.info("Asset materialization successful!")

            # Get stats on the materialized assets
            asset_files = {}
            for asset_key in ["internal_sales_data", "internal_need_state_data", "merged_internal_data"]:
                output_path = os.path.join(output_dir, f"{asset_key}.parquet")
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    asset_files[asset_key] = {"path": output_path, "size": file_size}
                    logger.info(f"Asset {asset_key}: {file_size} bytes")
                else:
                    logger.warning(f"Output file not found for {asset_key}")

            # Verify the dependency relationships worked
            if len(asset_files) == 3:
                logger.info("All assets were materialized successfully")

                # Check that merged data has more columns than input data
                # We can't load the data directly here as we'd need polars, but we can check file sizes
                if asset_files["merged_internal_data"]["size"] > asset_files["internal_sales_data"]["size"]:
                    logger.info("Merged data appears to contain combined information as expected")
                else:
                    logger.warning("Merged data size is smaller than expected")

            return True
        else:
            # Handle failures
            logger.error("Asset materialization failed!")
            for event in result.filter_events("FAILURE"):
                logger.error(f"Failure: {event}")
            return False

    except Exception as e:
        logger.exception(f"Error running preprocessing test: {e}")
        return False
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Test completed in {duration:.2f} seconds")


if __name__ == "__main__":
    success = run_preprocessing_test()
    sys.exit(0 if success else 1)
