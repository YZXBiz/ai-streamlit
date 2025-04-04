#!/usr/bin/env python
"""
A script to run just the preprocessing assets we've been fixing.
"""

import os
import sys
from datetime import datetime

from dagster import DagsterInstance, materialize

# Add the project directory to Python path if needed
sys.path.append(".")

# Import assets
from src.clustering.dagster.assets.preprocessing.internal import (
    internal_category_data,
    internal_need_state_data,
    internal_sales_data,
    merged_internal_data,
)
from src.clustering.dagster.definitions import create_definitions


def run_preprocessing():
    """Run the preprocessing assets separately from the full pipeline."""
    print(f"Starting preprocessing run at {datetime.now()}")

    # Create output directory
    output_dir = os.path.join("outputs", "preprocessing", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Get definitions and resources
    defs = create_definitions(env="dev")
    resources = defs.resources

    # Override io_manager to use our output directory
    resources["io_manager"] = resources["io_manager"].configured({"base_dir": output_dir})

    # Create an ephemeral Dagster instance
    instance = DagsterInstance.ephemeral()

    # Define the assets we want to materialize
    preprocessing_assets = [
        internal_sales_data,
        internal_need_state_data,
        merged_internal_data,
        internal_category_data,
    ]

    # Materialize the assets
    print("Materializing preprocessing assets...")
    result = materialize(
        preprocessing_assets,
        resources=resources,
        instance=instance,
        raise_on_error=False,
    )

    # Check the result
    if result.success:
        print("Preprocessing assets materialized successfully")

        # Print information about the materialized assets
        for event in result.get_asset_materialization_events():
            print(f"Materialized asset: {event.asset_key.path}")

        return True
    else:
        print("Preprocessing assets materialization failed")

        # Print failure information
        for event in result.get_step_events_by_kind("FAILURE"):
            print(f"Failure: {event}")

        return False


if __name__ == "__main__":
    success = run_preprocessing()
    sys.exit(0 if success else 1)
