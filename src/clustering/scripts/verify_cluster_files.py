#!/usr/bin/env python
"""
Verify cluster assignment files for debugging.

This script checks if the pickle files contain the expected data structure
for the merge_clusters asset and displays their contents for debugging.
"""

import pickle
import sys
from pathlib import Path
from typing import Any, List

import polars as pl


def load_pickle(file_path: str) -> Any:
    """Load data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The unpickled data
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading pickle from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data


def verify_cluster_file(file_path: str) -> None:
    """Verify if a file meets the requirements for the merge_clusters asset.
    
    Args:
        file_path: Path to the pickle file to verify
    """
    try:
        data = load_pickle(file_path)
        
        print(f"\nFile: {file_path}")
        print(f"Type: {type(data)}")
        
        # Check if it's a dict of DataFrames
        if isinstance(data, dict):
            print("Contents: Dictionary of DataFrames")
            for key, value in data.items():
                print(f"  Category: {key}")
                if isinstance(value, pl.DataFrame):
                    print(f"    Type: Polars DataFrame")
                    print(f"    Shape: {value.shape}")
                    print(f"    Columns: {value.columns}")
                    print(f"    Has STORE_NBR: {'STORE_NBR' in value.columns}")
                    print(f"    Cluster columns: {[col for col in value.columns if 'cluster' in col.lower()]}")
                    print(f"    First few rows:\n{value.head(3)}")
                else:
                    print(f"    Type: {type(value)} (not a Polars DataFrame)")
        
        # Check if it's a single DataFrame
        elif isinstance(data, pl.DataFrame):
            print("Contents: Single Polars DataFrame")
            print(f"Shape: {data.shape}")
            print(f"Columns: {data.columns}")
            print(f"Has STORE_NBR: {'STORE_NBR' in data.columns}")
            print(f"Cluster columns: {[col for col in data.columns if 'cluster' in col.lower()]}")
            print(f"First few rows:\n{data.head(3)}")
        else:
            print(f"Contents: Unknown type ({type(data)})")
            print(f"Data preview: {str(data)[:200]}")
            
    except Exception as e:
        print(f"Error verifying {file_path}: {str(e)}")


def main(files: List[str] = None) -> None:
    """Check specified pickle files or use default files if none specified.
    
    Args:
        files: List of files to check
    """
    base_dir = Path("/workspaces/testing-dagster/data")
    
    # Default files to check if none specified
    if not files:
        files = [
            base_dir / "external" / "cluster_assignments.pkl",
            base_dir / "external" / "external_data.pkl",
            base_dir / "external" / "processed_external_data.pkl",
            base_dir / "internal" / "cluster_assignments.pkl",  # Add internal cluster assignments
        ]
    
    # Display a summary of available files
    found_files = [str(f) for f in files if Path(f).exists()]
    missing_files = [str(f) for f in files if not Path(f).exists()]
    
    print(f"Found {len(found_files)} files:")
    for file in found_files:
        print(f"  - {file}")
    
    if missing_files:
        print(f"\nMissing {len(missing_files)} files:")
        for file in missing_files:
            print(f"  - {file}")
    
    print("\nStarting verification...")
    
    for file_path in files:
        try:
            verify_cluster_file(str(file_path))
            print("\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main() 