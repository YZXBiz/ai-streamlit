#!/usr/bin/env python
"""
A simple test script to verify the CSVReader works correctly.
"""

import sys

sys.path.append(".")

from src.clustering.io.readers.csv_reader import CSVReader


def test_csv_reader():
    """Test the CSVReader implementation."""
    print("Testing CSVReader...")

    # Create a CSVReader instance
    reader = CSVReader(path="data/raw/need_state.csv")

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
    test_csv_reader()
