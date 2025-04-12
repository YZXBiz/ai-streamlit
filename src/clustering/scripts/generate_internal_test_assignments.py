#!/usr/bin/env python
"""
Generate test internal cluster assignments for debugging.

This script creates a test version of the internal cluster assignments file
that matches the expected format with the required STORE_NBR column.
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


def generate_internal_test_assignments() -> None:
    """Generate test internal cluster assignments and save to pickle file."""
    # Create a simple DataFrame with store numbers and clusters
    # Ensure it has the STORE_NBR column that is required
    df = pl.DataFrame({
        "STORE_NBR": [100, 101, 102, 103, 104, 105],
        "Cluster": [1, 2, 1, 3, 2, 3],
        "Category": ["A", "B", "A", "C", "B", "C"],
        "Sales": [1200, 2200, 1800, 3200, 2800, 3800],
    })
    
    # Define the output paths
    base_dir = Path("/workspaces/testing-dagster/data")
    internal_dir = base_dir / "internal"
    
    # Create directories if they don't exist
    create_directory_if_not_exists(internal_dir)
    
    # Save the test data
    assignment_path = internal_dir / "cluster_assignments.pkl"
    with open(assignment_path, "wb") as f:
        pickle.dump(df, f)
    
    print(f"Generated test internal cluster assignments at: {assignment_path}")


if __name__ == "__main__":
    generate_internal_test_assignments()
    print("Internal test data generation complete.") 