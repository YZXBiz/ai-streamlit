#!/usr/bin/env python3
"""
Script to check the transformed data before merging.
"""

import os

import pandas as pd
import polars as pl

from clustering.core.sql_engine import SQL, DuckDB
from clustering.core.sql_templates import clean_need_state


def check_transformed_data():
    """Check the transformed data before merging."""

    # Create output directory
    os.makedirs("outputs/debug", exist_ok=True)

    print("Reading raw sales data...")
    sales_df = pl.read_parquet("data/raw/sales.parquet")
    print("Sales columns:", sales_df.columns)
    print("First row of sales data:", sales_df.head(1))

    print("\nReading raw need state data...")
    # Use pandas to read the CSV file, which handles quotes better
    need_state_pd = pd.read_csv("data/raw/need_state.csv")
    need_state_df = pl.from_pandas(need_state_pd)
    print("Need state columns:", need_state_df.columns)
    print("First row of need state data:", need_state_df.head(1))

    # Apply the clean_need_state transformation
    print("\nApplying clean_need_state transformation...")
    db = DuckDB()
    try:
        # Clean need state data using the SQL object
        clean_sql = clean_need_state(need_state_df)
        cleaned_need_state = db.query(clean_sql)

        print("Cleaned need state columns:", cleaned_need_state.columns)
        print("First row of cleaned need state data:", cleaned_need_state.head(1))

        # Save the transformed data for examination
        sales_df.write_parquet("outputs/debug/sales_debug.parquet")
        cleaned_need_state.write_parquet("outputs/debug/cleaned_need_state_debug.parquet")

        print("\nProposed SQL for merging:")

        # Create a simple SQL merge for testing
        test_merge_sql = SQL(
            """
            SELECT 
                s.*,
                n."NEED_STATE"
            FROM 
                $sales s
            JOIN 
                $need_state n
            ON 
                s."category_id" = n."CATEGORY_ID"
            LIMIT 5
            """,
            bindings={"sales": sales_df, "need_state": cleaned_need_state},
        )

        print(test_merge_sql.sql)
        print("\nAttempting test query with actual column names...")

        try:
            result = db.query(test_merge_sql)
            print("Test query succeeded!")
            print("Result columns:", result.columns)
            print("First row of result:", result.head(1))
        except Exception as e:
            print(f"Test query failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_transformed_data()
