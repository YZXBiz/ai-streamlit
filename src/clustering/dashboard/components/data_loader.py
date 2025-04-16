"""Data loader component for the clustering dashboard.

This module provides utilities to load data from Dagster assets for visualization.
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import pandas as pd
import polars as pl
import streamlit as st


@st.cache_data
def load_dagster_asset(
    asset_name: str, 
    base_path: Optional[str] = None
) -> Tuple[Union[pd.DataFrame, Dict[str, Any], None], str]:
    """Load a Dagster asset by name from storage.
    
    Args:
        asset_name: Name of the Dagster asset to load
        base_path: Base storage path, if None will look in standard locations
        
    Returns:
        Tuple with (loaded_data, path_loaded_from)
    """
    # Determine possible base paths
    if base_path is None:
        # Check in standard locations relative to project root
        base_paths = [
            "../storage/",  # Default Dagster storage location
            "../data/",     # Data directory
        ]
    else:
        base_paths = [base_path]
    
    # Try to load from possible paths
    full_paths = []
    for base in base_paths:
        # Try the exact name
        full_paths.append(os.path.join(base, asset_name))
        # Try as subdirectory with the standard name
        full_paths.append(os.path.join(base, asset_name, 'data.pkl'))
    
    # Try to load from each path
    for path in full_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                # Convert Polars DataFrame to pandas if needed
                if isinstance(data, pl.DataFrame):
                    return data.to_pandas(), path
                elif hasattr(data, 'to_pandas'):
                    return data.to_pandas(), path 
                elif isinstance(data, pd.DataFrame):
                    return data, path
                elif isinstance(data, dict) and any(isinstance(v, (pd.DataFrame, pl.DataFrame)) for v in data.values()):
                    # Handle dictionary of DataFrames
                    converted_data = {}
                    for k, v in data.items():
                        if isinstance(v, pl.DataFrame) or hasattr(v, 'to_pandas'):
                            converted_data[k] = v.to_pandas() if isinstance(v, pl.DataFrame) else v.to_pandas()
                        else:
                            converted_data[k] = v
                    return converted_data, path
                else:
                    return data, path
            except Exception as e:
                st.sidebar.error(f"Error loading {path}: {e}")
    
    return None, ""


def get_available_assets() -> list[str]:
    """Get a list of available Dagster assets.
    
    Returns:
        List of asset names that can be loaded
    """
    # Start with the merging assets which are most relevant for the dashboard
    primary_assets = [
        "cluster_reassignment",
        "merged_clusters",
        "optimized_merged_clusters",
        "merged_cluster_assignments"
    ]
    
    # Check storage directory for other assets
    storage_path = "../storage/"
    if os.path.exists(storage_path):
        try:
            storage_assets = [f for f in os.listdir(storage_path) 
                            if os.path.isfile(os.path.join(storage_path, f))]
            # Add any assets not already in primary_assets
            for asset in storage_assets:
                if asset not in primary_assets:
                    primary_assets.append(asset)
        except Exception:
            pass
    
    return primary_assets


def extract_features(data: pd.DataFrame) -> list[str]:
    """Extract features from DataFrame that aren't cluster or ID columns.
    
    Args:
        data: DataFrame with features and cluster columns
        
    Returns:
        List of feature column names
    """
    # Exclude common non-feature columns
    exclude_patterns = ['cluster', 'store', 'id', 'index', 'category']
    
    # Get columns that don't match exclude patterns
    feature_cols = [
        col for col in data.columns
        if not any(pattern in col.lower() for pattern in exclude_patterns)
    ]
    
    return feature_cols


def extract_cluster_columns(data: pd.DataFrame) -> list[str]:
    """Extract cluster assignment columns from DataFrame.
    
    Args:
        data: DataFrame with features and cluster columns
        
    Returns:
        List of cluster column names
    """
    return [col for col in data.columns if 'cluster' in col.lower()]


def create_feature_subset(
    data: pd.DataFrame, 
    cluster_col: str,
    n_features: int = 10
) -> pd.DataFrame:
    """Create a subset with the most relevant features for visualization.
    
    Args:
        data: DataFrame with features and cluster columns
        cluster_col: Name of the cluster column
        n_features: Number of features to include
        
    Returns:
        DataFrame with selected features
    """
    # Get all potential feature columns
    feature_cols = extract_features(data)
    
    if len(feature_cols) <= n_features:
        # If we have fewer features than requested, return all
        return data
    
    # Select the first n features
    # In a real implementation, you could use feature importance
    selected_features = feature_cols[:n_features]
    
    # Return data with cluster column and selected features
    return data[[cluster_col] + selected_features] 