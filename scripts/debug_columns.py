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

# Removed SQL and DuckDB imports
# from src.clustering.core.sql_engine import SQL, DuckDB
# from src.clustering.core.sql_templates import clean_need_state
from src.clustering.dagster.assets.preprocessing.internal import (
    internal_need_state_data,
    internal_sales_data,
)
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

    # Clean need state data using Polars instead of SQL
    print("\nCleaning need state data...")
    # Replace SQL template with direct Polars operations
    cleaned_need_state = need_state_df.filter(pl.col("category_id").is_not_null()).with_columns(
        [
            pl.col("category_id").alias("CATEGORY_ID"),
            pl.col("need_state_id").alias("NEED_STATE_ID"),
            pl.col("need_state_name").alias("NEED_STATE_NAME"),
            pl.col("need_state_description").alias("NEED_STATE_DESCRIPTION"),
            pl.col("category_id").cast(pl.Int64).alias("PRODUCT_ID"),
            pl.col("need_state_name").str.to_uppercase().alias("NEED_STATE"),
            pl.col("need_state_description").alias("CATEGORY"),
            pl.lit("Unknown").alias("CDT"),
            pl.lit("Unknown").alias("ATTRIBUTE_1"),
            pl.lit("Unknown").alias("ATTRIBUTE_2"),
            pl.lit("Unknown").alias("ATTRIBUTE_3"),
            pl.lit("Unknown").alias("ATTRIBUTE_4"),
            pl.lit("Unknown").alias("ATTRIBUTE_5"),
            pl.lit("Unknown").alias("ATTRIBUTE_6"),
            pl.col("need_state_name").alias("PLANOGRAM_DSC"),
            pl.col("need_state_id").alias("PLANOGRAM_NBR"),
            pl.lit(False).alias("NEW_ITEM"),
            pl.lit(False).alias("TO_BE_DROPPED"),
        ]
    )
    print("Cleaned need state columns:", cleaned_need_state.columns)

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

        # Now try a test merge using Polars instead of DuckDB
        print("\n=== Testing Merge Query ===")
        try:
            # Using Polars for join
            result = (
                sales_processed.join(
                    need_state_processed, left_on="CAT_DSC", right_on="CATEGORY_ID", how="inner"
                )
                .select(
                    [
                        pl.col("SKU_NBR"),
                        pl.col("CAT_DSC"),
                        pl.col("PRODUCT_ID"),
                        pl.col("NEED_STATE"),
                    ]
                )
                .limit(5)
            )
            print("Test merge succeeded!")
            print("Result columns:", result.columns)
            print("First few rows of result:", result.head(3))
        except Exception as e:
            print(f"Test merge failed: {e}")

            # Try another approach
            print("\nTrying alternative merge...")
            try:
                # Using Polars with different join syntax
                alt_result = sales_processed.join(
                    need_state_processed, left_on="CAT_DSC", right_on="CATEGORY_ID", how="inner"
                ).limit(5)
                print("Alternative merge succeeded!")
                print("Result columns:", alt_result.columns)
                print("First few rows of result:", alt_result.head(3))
            except Exception as e2:
                print(f"Alternative merge also failed: {e2}")
    else:
        print("Asset materialization failed!")


if __name__ == "__main__":
    debug_columns()
