#!/usr/bin/env python
"""
A simplified script that runs just the compute function directly without IO managers.
"""

import sys
from pathlib import Path

import polars as pl

# Setup path for imports
sys.path.append(".")

from src.clustering.core import schemas
from src.clustering.io.readers.parquet_reader import ParquetReader


def run_direct():
    """Run the internal_sales_data compute function directly."""
    print("Running internal_sales_data directly...")

    # Read the sales data
    print("Reading sales data...")
    reader = ParquetReader(path="data/internal/sales.parquet")
    sales_data_raw = reader.read()

    # Convert to Polars if needed
    if not isinstance(sales_data_raw, pl.DataFrame):
        sales_data = pl.from_pandas(sales_data_raw)
    else:
        sales_data = sales_data_raw

    # Validate the sales data
    sales_data_validated = schemas.InputsSalesSchema.check(sales_data)

    print(f"Successfully processed sales data with shape: {sales_data_validated.shape}")
    print(f"Columns: {sales_data_validated.columns}")
    print(f"First few rows:\n{sales_data_validated.head(3)}")

    # Create output directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save the data to a parquet file
    output_path = output_dir / "internal_sales_processed.parquet"
    print(f"Saving processed data to {output_path}")
    sales_data_validated.write_parquet(output_path)

    print("Processing complete!")
    return sales_data_validated


if __name__ == "__main__":
    run_direct()
