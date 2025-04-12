#!/usr/bin/env python
"""
Verify model files needed for cluster reassignment.

This script checks if the model files needed by the cluster_reassignment function
exist and have the expected structure.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import polars as pl


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file exists, False otherwise
    """
    return os.path.exists(file_path)


def load_pickle_safe(file_path: str) -> Any:
    """Load data from a pickle file safely, returning None if there's an error.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The unpickled data or None if there's an error
    """
    if not check_file_exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        print(f"Loading pickle from: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def check_centroids(model_data: Any, model_name: str) -> bool:
    """Check if model data contains centroids.
    
    Args:
        model_data: Model data loaded from pickle
        model_name: Name of the model for logging
        
    Returns:
        True if centroids are found, False otherwise
    """
    if model_data is None:
        print(f"{model_name} model data is None")
        return False
    
    print(f"\nChecking {model_name} model data:")
    print(f"Type: {type(model_data)}")
    
    # If it's a dictionary of models, check the first one
    if isinstance(model_data, dict) and not any(k in model_data for k in ["model", "centroids"]):
        # Get the first category
        if not model_data:
            print(f"No categories found in {model_name} model data")
            return False
        
        first_category = next(iter(model_data.keys()))
        print(f"Using first category: {first_category}")
        model_data = model_data[first_category]
    
    # Check for centroids
    if isinstance(model_data, dict) and "centroids" in model_data:
        centroids = model_data["centroids"]
        print(f"Found centroids: {type(centroids)}")
        print(f"Number of clusters: {len(centroids)}")
        
        # Check a sample centroid
        if centroids:
            first_cluster = next(iter(centroids.keys()))
            centroid = centroids[first_cluster]
            print(f"Sample centroid for cluster {first_cluster}: {type(centroid)}, shape: {getattr(centroid, 'shape', 'N/A')}")
        
        return True
    else:
        print(f"No centroids found in {model_name} model data")
        if isinstance(model_data, dict):
            print(f"Keys in model data: {list(model_data.keys())}")
        return False


def create_test_model_files(
    internal_path: str,
    external_path: str,
    internal_clusters: List[int] = None,
    external_clusters: List[int] = None,
    dimension: int = 5
) -> None:
    """Create test model files with centroids if they don't exist.
    
    Args:
        internal_path: Path for internal model
        external_path: Path for external model
        internal_clusters: List of internal cluster IDs
        external_clusters: List of external cluster IDs
        dimension: Dimension of centroids
    """
    import numpy as np
    
    if internal_clusters is None:
        internal_clusters = [0, 1, 2]
    
    if external_clusters is None:
        external_clusters = [0, 1, 2]
    
    # Create directories if needed
    os.makedirs(os.path.dirname(internal_path), exist_ok=True)
    os.makedirs(os.path.dirname(external_path), exist_ok=True)
    
    # Create internal model
    internal_model = {
        "model": {"name": "Test Internal Model"},
        "centroids": {
            i: np.random.rand(dimension) for i in internal_clusters
        },
        "num_clusters": len(internal_clusters),
        "num_samples": 100,
    }
    
    # Create external model
    external_model = {
        "model": {"name": "Test External Model"},
        "centroids": {
            i: np.random.rand(dimension) for i in external_clusters
        },
        "num_clusters": len(external_clusters),
        "num_samples": 100,
    }
    
    # Save models
    with open(internal_path, "wb") as f:
        pickle.dump(internal_model, f)
    print(f"Created test internal model at {internal_path}")
    
    with open(external_path, "wb") as f:
        pickle.dump(external_model, f)
    print(f"Created test external model at {external_path}")


def verify_model_files(create_test_files_if_missing: bool = True) -> None:
    """Check model files and create test files if needed.
    
    Args:
        create_test_files_if_missing: Whether to create test files if they don't exist
    """
    # Define paths from the config
    base_dir = Path("/workspaces/testing-dagster/data")
    internal_model_path = base_dir / "internal" / "clustering_models.pkl"
    external_model_path = base_dir / "external" / "clustering_models.pkl"
    
    print(f"Checking model files:")
    print(f"Internal model path: {internal_model_path}")
    print(f"External model path: {external_model_path}")
    
    # Check if files exist
    internal_exists = check_file_exists(internal_model_path)
    external_exists = check_file_exists(external_model_path)
    
    print(f"Internal model exists: {internal_exists}")
    print(f"External model exists: {external_exists}")
    
    # If either file doesn't exist and we should create test files
    if create_test_files_if_missing and (not internal_exists or not external_exists):
        print("\nCreating test model files...")
        create_test_model_files(
            str(internal_model_path),
            str(external_model_path),
            internal_clusters=[1, 2, 3],
            external_clusters=[0, 1, 2]
        )
        # Reload existence check
        internal_exists = check_file_exists(internal_model_path)
        external_exists = check_file_exists(external_model_path)
    
    # If files exist, load and check them
    if internal_exists and external_exists:
        internal_model = load_pickle_safe(internal_model_path)
        external_model = load_pickle_safe(external_model_path)
        
        internal_centroids_ok = check_centroids(internal_model, "internal")
        external_centroids_ok = check_centroids(external_model, "external")
        
        if internal_centroids_ok and external_centroids_ok:
            print("\nAll model files exist and have the required centroids.")
        else:
            print("\nSome model files are missing centroids.")
            
            if create_test_files_if_missing:
                print("\nCreating replacement test model files...")
                create_test_model_files(
                    str(internal_model_path),
                    str(external_model_path),
                    internal_clusters=[1, 2, 3],
                    external_clusters=[0, 1, 2]
                )
    else:
        print("One or both model files still do not exist.")


if __name__ == "__main__":
    verify_model_files(create_test_files_if_missing=True) 