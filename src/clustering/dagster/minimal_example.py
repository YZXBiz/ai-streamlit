"""Minimal example for testing Dagster implementation with SQL engine."""

import dagster as dg
import polars as pl

from clustering.core.sql_engine import SQL, DuckDB


@dg.asset
def sample_data() -> pl.DataFrame:
    """Generate sample data for testing SQL transformations."""
    return pl.DataFrame(
        {
            "SKU_NBR": ["SKU001", "SKU002", "SKU003"],
            "STORE_NBR": ["S001", "S002", "S003"],
            "TOTAL_SALES": [100, 200, 300],
        }
    )


@dg.asset
def processed_data(sample_data: pl.DataFrame) -> pl.DataFrame:
    """Process sample data using SQL engine.

    Demonstrates how to use the DuckDB engine with SQL templates.

    Args:
        sample_data: Input sales data

    Returns:
        DataFrame with processed data
    """
    db = DuckDB()
    try:
        sql_obj = SQL(
            """
            SELECT
                "SKU_NBR",
                "STORE_NBR",
                "TOTAL_SALES",
                "TOTAL_SALES" * 0.1 AS "TAX_AMOUNT"
            FROM $data
            """,
            bindings={"data": sample_data},
        )
        return db.query(sql_obj)
    finally:
        db.close()


# Define a simple job
minimal_job = dg.define_asset_job("minimal_job", selection=[processed_data])

# Create Dagster definitions
defs = dg.Definitions(assets=[sample_data, processed_data], jobs=[minimal_job])


if __name__ == "__main__":
    # Run the Dagster webserver directly from this file
    dg.webserver.run_webserver(
        workspace=dg.workspace.LoadableTargetWorkspace(
            loadable_target=dg.workspace.LoadableTarget(
                attribute="defs", python_module="clustering.dagster.minimal_example"
            )
        )
    )
