#!/usr/bin/env python
"""
Ensure store overlap between internal and external cluster files.

This script makes sure there are common store IDs between the internal and external
cluster assignment files to avoid join failures in the merged_clusters asset.
"""

import pickle
from pathlib import Path

import polars as pl


def ensure_store_overlap() -> None:
    """Update cluster files to ensure store ID overlap."""
    # Define paths
    internal_path = Path("/workspaces/testing-dagster/data/internal/cluster_assignments.pkl")
    external_path = Path("/workspaces/testing-dagster/data/external/cluster_assignments.pkl")
    
    # Check if files exist
    if not internal_path.exists() or not external_path.exists():
        print("Error: One or both cluster files do not exist")
        return
    
    # Load data
    with open(internal_path, "rb") as f:
        internal_data = pickle.load(f)
    
    with open(external_path, "rb") as f:
        external_data = pickle.load(f)
    
    # Check if both are DataFrames
    if not isinstance(internal_data, pl.DataFrame) or not isinstance(external_data, pl.DataFrame):
        print("Error: One or both files do not contain Polars DataFrames")
        return
    
    # Get store IDs
    internal_stores = set(internal_data.select("STORE_NBR").to_series().to_list())
    external_stores = set(external_data.select("STORE_NBR").to_series().to_list())
    
    # Find common stores
    common_stores = internal_stores.intersection(external_stores)
    
    print(f"Internal stores: {internal_stores}")
    print(f"External stores: {external_stores}")
    print(f"Common stores: {common_stores}")
    
    # If no common stores, add some
    if not common_stores:
        print("No common stores found. Adding overlapping stores...")
        
        # Choose stores to make common
        internal_store = list(internal_stores)[0] if internal_stores else 999
        external_store = list(external_stores)[0] if external_stores else 888
        
        # Add internal store to external data
        if internal_store not in external_stores:
            # Get a sample row to copy format
            sample_row = external_data.row(0, named=True)
            
            # Create new row with internal store ID
            new_row = {col: sample_row[col] for col in external_data.columns}
            new_row["STORE_NBR"] = internal_store
            new_row["Cluster"] = 0  # Default cluster
            
            # Add to external data
            external_data = pl.concat([external_data, pl.DataFrame([new_row])])
            
            print(f"Added store {internal_store} to external data")
        
        # Add external store to internal data if needed
        if external_store not in internal_stores and external_store != internal_store:
            # Get a sample row to copy format
            sample_row = internal_data.row(0, named=True)
            
            # Create new row with external store ID
            new_row = {col: sample_row[col] for col in internal_data.columns}
            new_row["STORE_NBR"] = external_store
            new_row["Cluster"] = 0  # Default cluster
            
            # Add to internal data
            internal_data = pl.concat([internal_data, pl.DataFrame([new_row])])
            
            print(f"Added store {external_store} to internal data")
        
        # Save updated data
        print("Saving updated files...")
        
        with open(internal_path, "wb") as f:
            pickle.dump(internal_data, f)
        
        with open(external_path, "wb") as f:
            pickle.dump(external_data, f)
        
        print("Files updated successfully")
    else:
        print(f"Found {len(common_stores)} common stores. No updates needed.")


if __name__ == "__main__":
    ensure_store_overlap() 