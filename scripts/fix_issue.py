#!/usr/bin/env python
"""
A script to fix the issues with the Dagster pipeline by materializing assets in order.
"""

import os
import sys

from dagster import DagsterInstance, materialize

# Import the Dagster definitions
sys.path.append(".")
from src.clustering.dagster.assets import (
    product_category_mapping,
    raw_sales_data,
    sales_with_categories,
)
from src.clustering.dagster.definitions import create_definitions


def fix_issue():
    """Fix the issue by materializing all assets together."""
    # Create definitions with dev environment
    defs = create_definitions(env="dev")

    # Get an ephemeral instance
    instance = DagsterInstance.ephemeral()

    # Get all resources
    resources = defs.resources

    # Make sure storage directory exists
    os.makedirs("storage", exist_ok=True)

    # Materialize all assets together
    print("Materializing all assets together...")
    try:
        result = materialize(
            [raw_sales_data, product_category_mapping, sales_with_categories],
            resources=resources,
            instance=instance,
            raise_on_error=False,
        )

        if result.success:
            print("Materialization of all assets successful!")
            return result
        else:
            print("Materialization failed!")
            # Print the failure information from the result
            for event in result.all_node_events:
                if event.is_step_failure:
                    error_msg = event.message
                    if hasattr(event, "event_specific_data"):
                        if hasattr(event.event_specific_data, "error"):
                            error_msg = f"{error_msg}: {event.event_specific_data.error}"
                    print(f"Error: {error_msg}")
            return result
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    fix_issue()
