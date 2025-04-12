#!/usr/bin/env python
"""
Generate test cluster assignments for debugging.

This script creates a minimal test version of the cluster assignments file
that matches the expected format for the merged_clusters asset.
"""

import os
import pickle
from pathlib import Path

import polars as pl


def create_directory_if_not_exists(directory_path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def generate_test_cluster_assignments() -> None:
    """Generate test cluster assignments and save to pickle file."""
    # Create a simple DataFrame with store numbers and clusters
    df = pl.DataFrame({
        "STORE_NBR": [100, 101, 102, 103, 104, 105],
        "Cluster": [0, 1, 0, 2, 1, 2],
        "SALES": [1000, 2000, 1500, 3000, 2500, 3500],
    })
    
    # Define the output paths
    base_dir = Path("/workspaces/testing-dagster/data")
    external_dir = base_dir / "external"
    
    # Create directories if they don't exist
    create_directory_if_not_exists(external_dir)
    
    # Save the test data
    assignment_path = external_dir / "cluster_assignments.pkl"
    with open(assignment_path, "wb") as f:
        pickle.dump(df, f)
    
    print(f"Generated test cluster assignments at: {assignment_path}")


if __name__ == "__main__":
    generate_test_cluster_assignments()
    print("Test data generation complete.") 