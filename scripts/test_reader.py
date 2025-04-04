#!/usr/bin/env python
"""
A simple test script to verify the ParquetReader works correctly.
"""

import sys

sys.path.append(".")

from src.clustering.io.readers.parquet_reader import ParquetReader


def test_parquet_reader():
    """Test the ParquetReader implementation."""
    print("Testing ParquetReader...")

    # Create a ParquetReader instance
    reader = ParquetReader(path="data/raw/internal_sales.parquet")

    try:
        # Read the data
        print("Reading data...")
        data = reader.read()
        print(f"Successfully read data with shape: {data.shape}")
        print(f"Columns: {data.columns}")
        print(f"First few rows:\n{data.head(3)}")
        return True
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_parquet_reader()
