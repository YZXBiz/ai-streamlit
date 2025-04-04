#!/usr/bin/env python3
"""
Script to debug column names at different stages of the pipeline.
"""

import os
import sys

import pandas as pd
import polars as pl
from dagster import DagsterInstance, materialize

# Add src directory to Python path if needed
sys.path.append(".")

from src.clustering.core.sql_engine import SQL, DuckDB
from src.clustering.core.sql_templates import clean_need_state
from src.clustering.dagster.assets.preprocessing.internal import internal_need_state_data, internal_sales_data
from src.clustering.dagster.resources.data_io import data_reader
from src.clustering.dagster.resources.io_manager import clustering_io_manager


def debug_columns():
    """Debug column names at different stages of the pipeline."""

    # Create output directory
    output_dir = "outputs/debug"
    os.makedirs(output_dir, exist_ok=True)

    print("=== Direct Data Reading ===")

    # Read sales data directly
    print("\nReading sales data directly...")
    sales_df = pl.read_parquet("data/raw/sales.parquet")
    print("Original sales columns:", sales_df.columns)

    # Read need state data directly
    print("\nReading need state data directly...")
    need_state_pd = pd.read_csv("data/raw/need_state.csv")
    need_state_df = pl.from_pandas(need_state_pd)
    print("Original need state columns:", need_state_df.columns)

    # Clean need state data
    db = DuckDB()
    try:
        print("\nCleaning need state data...")
        clean_sql = clean_need_state(need_state_df)
        cleaned_need_state = db.query(clean_sql)
        print("Cleaned need state columns:", cleaned_need_state.columns)
    finally:
        db.close()

    print("\n=== Using Dagster Assets ===")

    # Create a temporary output directory for this test
    dagster_output_dir = os.path.join(output_dir, "dagster")
    os.makedirs(dagster_output_dir, exist_ok=True)

    # Define resources
    resources = {
        "io_manager": clustering_io_manager.configured({"base_dir": dagster_output_dir}),
        "input_sales_reader": data_reader.configured(
            {"kind": "ParquetReader", "config": {"path": "data/raw/sales.parquet"}}
        ),
        "input_need_state_reader": data_reader.configured(
            {"kind": "CSVReader", "config": {"path": "data/raw/need_state.csv"}}
        ),
        "config": None,  # Not needed for this test
    }

    # Create an ephemeral Dagster instance
    instance = DagsterInstance.ephemeral()

    # Materialize the assets
    print("\nMaterializing assets...")
    result = materialize(
        [internal_sales_data, internal_need_state_data],
        resources=resources,
        instance=instance,
        raise_on_error=True,
    )

    if result.success:
        print("\nAssets materialized successfully!")

        # Read the materialized files
        sales_path = os.path.join(dagster_output_dir, "internal_sales_data.parquet")
        need_state_path = os.path.join(dagster_output_dir, "internal_need_state_data.parquet")

        if os.path.exists(sales_path):
            sales_processed = pl.read_parquet(sales_path)
            print("\nProcessed sales columns:", sales_processed.columns)
            print("First few rows of processed sales:", sales_processed.head(3))

        if os.path.exists(need_state_path):
            need_state_processed = pl.read_parquet(need_state_path)
            print("\nProcessed need state columns:", need_state_processed.columns)
            print("First few rows of processed need state:", need_state_processed.head(3))

        # Now try a test merge
        print("\n=== Testing Merge Query ===")
        db = DuckDB()
        try:
            test_merge_sql = SQL(
                """
                SELECT
                    s."SKU_NBR",
                    s."CAT_DSC",
                    n."PRODUCT_ID",
                    n."NEED_STATE"
                FROM
                    $sales s
                JOIN
                    $need_state n
                ON
                    s."CAT_DSC" = CAST(n."CATEGORY_ID" AS VARCHAR)
                LIMIT 5
                """,
                bindings={"sales": sales_processed, "need_state": need_state_processed},
            )

            try:
                result = db.query(test_merge_sql)
                print("Test merge succeeded!")
                print("Result columns:", result.columns)
                print("First few rows of result:", result.head(3))
            except Exception as e:
                print(f"Test merge failed: {e}")

                # Try another approach
                print("\nTrying alternative merge...")
                alt_merge_sql = SQL(
                    """
                    SELECT * FROM $sales s, $need_state n 
                    WHERE s."CAT_DSC" = n."CATEGORY_ID"::VARCHAR
                    LIMIT 5
                    """,
                    bindings={"sales": sales_processed, "need_state": need_state_processed},
                )

                try:
                    alt_result = db.query(alt_merge_sql)
                    print("Alternative merge succeeded!")
                    print("Result columns:", alt_result.columns)
                    print("First few rows of result:", alt_result.head(3))
                except Exception as e2:
                    print(f"Alternative merge also failed: {e2}")
        finally:
            db.close()
    else:
        print("Asset materialization failed!")


if __name__ == "__main__":
    debug_columns()
