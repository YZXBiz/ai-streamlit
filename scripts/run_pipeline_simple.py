#!/usr/bin/env python
"""
A simplified script to run specific assets from the Dagster pipeline.
"""

import sys

from dagster import DagsterInstance, materialize

# Import the Dagster definitions
sys.path.append(".")
from src.clustering.dagster.assets import internal_sales_data
from src.clustering.dagster.definitions import create_definitions


def run_simple():
    """Run a simple materialize operation for just one asset."""
    # Create definitions with dev environment
    defs = create_definitions(env="dev")

    # Get an ephemeral instance
    instance = DagsterInstance.ephemeral()

    # Get all resources
    resources = defs.resources

    # Materialize just the internal_sales_data asset
    print("Materializing internal_sales_data asset...")
    try:
        result = materialize(
            [internal_sales_data],
            resources=resources,
            instance=instance,
            raise_on_error=False,
        )

        if result.success:
            print("Materialization successful!")
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
    run_simple()
