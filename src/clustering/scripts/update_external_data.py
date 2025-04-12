#!/usr/bin/env python
"""
Update external_data.pkl file to add Cluster column.

This script loads the existing external_data.pkl file, adds a Cluster column,
and saves it back to the same file.
"""

import pickle
from pathlib import Path

import polars as pl


def update_external_data_with_cluster() -> None:
    """Update external_data.pkl file to add Cluster column."""
    # Define the path
    file_path = Path("/workspaces/testing-dagster/data/external/external_data.pkl")
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Loading data from {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, pl.DataFrame):
        print(f"Error: File {file_path} does not contain a Polars DataFrame")
        return
    
    print(f"Original data columns: {data.columns}")
    
    # Check if Cluster column already exists
    if any(col.lower() == "cluster" for col in data.columns):
        print("Cluster column already exists, no update needed")
        return
    
    # Add Cluster column based on STORE_NBR
    if "STORE_NBR" in data.columns:
        # Assign clusters based on store number modulo 3 (for testing purposes)
        data = data.with_columns(
            (pl.col("STORE_NBR") % 3).alias("Cluster")
        )
        
        print(f"Updated data columns: {data.columns}")
        print(f"Data preview:\n{data.head(3)}")
        
        # Save the updated DataFrame
        print(f"Saving updated data to {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        print("Update completed successfully")
    else:
        print("Error: STORE_NBR column not found in the data")


if __name__ == "__main__":
    update_external_data_with_cluster() 