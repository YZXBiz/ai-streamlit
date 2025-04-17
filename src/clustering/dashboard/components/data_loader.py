"""Data loader component for the clustering dashboard.

This module provides utilities to load data from Dagster assets for visualization.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union
import json

import pandas as pd
import polars as pl
import streamlit as st
from dagster import DagsterInstance, AssetKey
from dagster_graphql import DagsterGraphQLClient
# from dagster_graphql.client.mutations import execute_partition_set_materialize_mutation
from dagster._core.instance import InstanceRef
import requests


@st.cache_data
def load_dagster_asset(
    asset_name: str, 
    base_path: Optional[str] = None
) -> tuple[Union[pd.DataFrame, dict[str, Any], None], str]:
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


def get_available_assets() -> list[dict[str, Any]]:
    """
    Retrieves a list of available assets from Dagster.
    
    Returns:
        List of assets with their metadata
    """
    # Create a Dagster GraphQL client
    client = DagsterGraphQLClient("http://dagster-webserver:3000")
    
    try:
        # Query all available assets
        result = client.get_assets()
        assets = []
        
        if result and result.get("assetsOrError", {}).get("__typename") == "AssetConnection":
            nodes = result.get("assetsOrError", {}).get("nodes", [])
            
            for node in nodes:
                asset_key = node.get("key", {}).get("path", [])
                if asset_key:
                    key_string = ".".join(asset_key)
                    
                    # Get metadata about this asset
                    metadata = {
                        "key": key_string,
                        "description": node.get("description", ""),
                        "path": asset_key,
                        "last_materialized": node.get("lastMaterialization", {}).get("timestamp")
                    }
                    
                    assets.append(metadata)
        
        return assets
    except Exception as e:
        st.error(f"Error connecting to Dagster: {str(e)}")
        return []
    

def load_from_dagster(asset_key_str: str) -> Optional[pd.DataFrame]:
    """
    Loads a dataset from Dagster using the asset key.
    
    Args:
        asset_key_str: The asset key string in format "path.to.asset"
        
    Returns:
        DataFrame containing the loaded data or None if loading failed
    """
    try:
        # Create asset key from string
        key_parts = asset_key_str.split(".")
        asset_key = AssetKey(key_parts)
        
        # Create a Dagster GraphQL client
        client = DagsterGraphQLClient("http://dagster-webserver:3000")
        
        # Get the latest materialization of this asset
        result = client.get_asset_materialization_events(asset_key, limit=1)
        
        if not result or not result.get("assetOrError", {}).get("materializationEvents"):
            st.error(f"No materializations found for asset {asset_key_str}")
            return None
        
        # Get the storage path from the materialization metadata
        materialization = result["assetOrError"]["materializationEvents"][0]
        metadata_entries = materialization.get("metadataEntries", [])
        
        storage_path = None
        for entry in metadata_entries:
            if entry.get("label") == "path":
                storage_path = entry.get("path", {}).get("path")
                break
        
        if not storage_path:
            st.error(f"No storage path found for asset {asset_key_str}")
            return None
        
        # Load the data based on file extension
        if storage_path.endswith(".csv"):
            return pd.read_csv(storage_path)
        elif storage_path.endswith(".parquet"):
            return pd.read_parquet(storage_path)
        elif storage_path.endswith(".json"):
            return pd.read_json(storage_path)
        else:
            st.error(f"Unsupported file format for asset {asset_key_str}")
            return None
            
    except Exception as e:
        st.error(f"Error loading asset {asset_key_str}: {str(e)}")
        return None


def load_local_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a local dataset from the specified file path.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        DataFrame containing the loaded data or None if loading failed
    """
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return None


def get_sample_data_paths() -> list[str]:
    """
    Returns a list of available sample datasets.
    
    Returns:
        List of file paths to sample datasets
    """
    # Define the directory where sample datasets are stored
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    if not os.path.exists(sample_dir):
        return []
    
    # Get all CSV, Parquet, and JSON files
    files = []
    for root, _, filenames in os.walk(sample_dir):
        for filename in filenames:
            if filename.endswith((".csv", ".parquet", ".json")):
                files.append(os.path.join(root, filename))
    
    return files


def data_source_selector() -> tuple[Optional[pd.DataFrame], str, str]:
    """
    Provides an interface for selecting data sources and loading datasets.
    
    Returns:
        Tuple containing (loaded_dataframe, source_type, source_name)
    """
    st.write("## Data Source")
    
    source_type = st.radio(
        "Select data source",
        ["Dagster Assets", "Sample Datasets", "Upload File"],
        horizontal=True
    )
    
    df = None
    source_name = ""
    
    if source_type == "Dagster Assets":
        assets = get_available_assets()
        
        if not assets:
            st.warning("No assets found in Dagster or couldn't connect to Dagster.")
            st.info("Please select another data source or check your Dagster connection.")
        else:
            # Format asset options
            asset_options = {
                f"{asset['key']} - {asset['description'][:30] + '...' if len(asset['description']) > 30 else asset['description']}": 
                asset['key'] for asset in assets
            }
            
            selected_asset_display = st.selectbox(
                "Select a Dagster asset",
                options=list(asset_options.keys())
            )
            
            if selected_asset_display:
                selected_asset_key = asset_options[selected_asset_display]
                source_name = selected_asset_key
                
                with st.spinner(f"Loading asset {selected_asset_key}..."):
                    df = load_from_dagster(selected_asset_key)
                
                if df is not None:
                    st.success(f"Successfully loaded asset: {selected_asset_key}")
    
    elif source_type == "Sample Datasets":
        sample_files = get_sample_data_paths()
        
        if not sample_files:
            st.warning("No sample datasets found.")
            st.info("Please select another data source or add sample datasets to the 'data' directory.")
        else:
            # Format filenames for display
            file_options = {os.path.basename(f): f for f in sample_files}
            
            selected_file_display = st.selectbox(
                "Select a sample dataset",
                options=list(file_options.keys())
            )
            
            if selected_file_display:
                selected_file_path = file_options[selected_file_display]
                source_name = selected_file_display
                
                with st.spinner(f"Loading {selected_file_display}..."):
                    df = load_local_dataset(selected_file_path)
                
                if df is not None:
                    st.success(f"Successfully loaded: {selected_file_display}")
    
    elif source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a dataset file", 
            type=["csv", "parquet", "json"]
        )
        
        if uploaded_file is not None:
            source_name = uploaded_file.name
            
            try:
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    # Determine file type and load accordingly
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".parquet"):
                        df = pd.read_parquet(uploaded_file)
                    elif uploaded_file.name.endswith(".json"):
                        df = pd.read_json(uploaded_file)
                
                if df is not None:
                    st.success(f"Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # If data is loaded, show a preview
    if df is not None:
        with st.expander("Data Preview", expanded=True):
            st.write(f"### Preview of {source_name}")
            st.dataframe(df.head(5), use_container_width=True)
            
            st.write("### Dataset Info")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    return df, source_type, source_name


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


def get_available_dagster_assets() -> list[str]:
    """
    Get a list of all available Dagster assets.
    
    Returns:
        List of asset names that can be loaded.
    """
    # Default location for Dagster asset storage
    dagster_root = Path(os.environ.get("DAGSTER_HOME", "/opt/dagster/storage"))
    assets_dir = dagster_root / "storage"
    
    # Check if directory exists
    if not assets_dir.exists():
        # Try alternative locations
        alt_locations = [
            Path("./data/assets"),
            Path("./output"),
            Path("./assets"),
        ]
        
        for loc in alt_locations:
            if loc.exists():
                assets_dir = loc
                break
        else:
            # If no location found, return empty list
            return []
    
    # List all potential asset files
    asset_files = []
    for ext in [".csv", ".parquet", ".json", ".pkl", ".feather"]:
        asset_files.extend(list(assets_dir.glob(f"**/*{ext}")))
    
    # Extract asset names without extension
    asset_names = [f.stem for f in asset_files]
    
    return sorted(asset_names)


def load_asset_data(asset_name: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Load data from a Dagster asset.
    
    Args:
        asset_name: Name of the asset to load
        
    Returns:
        Tuple of (DataFrame or None if loading failed, error message if any)
    """
    try:
        # Initialize Dagster instance
        instance = DagsterInstance.get()
        
        # Parse asset key from name
        from dagster import AssetKey
        asset_key = AssetKey.from_string(asset_name)
        
        # Get the latest materialization event for this asset
        latest_materialization = instance.get_latest_materialization_event(asset_key)
        
        if not latest_materialization:
            return None, f"Asset {asset_name} has not been materialized yet."
        
        # Get the storage path from metadata
        storage_metadata = latest_materialization.event_specific_data.materialization.metadata.get('storage_path')
        
        if not storage_metadata:
            return None, f"Asset {asset_name} does not have storage path metadata."
        
        storage_path = str(storage_metadata.value)
        
        # Load the data based on file extension
        if storage_path.endswith('.csv'):
            df = pd.read_csv(storage_path)
        elif storage_path.endswith('.parquet'):
            df = pd.read_parquet(storage_path)
        elif storage_path.endswith('.json'):
            df = pd.read_json(storage_path)
        else:
            return None, f"Unsupported file format: {storage_path}"
        
        return df, ""
    except Exception as e:
        # Fallback to loading sample datasets if we can't load from Dagster
        if asset_name == 'iris_dataset':
            return pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'), ""
        elif asset_name == 'wine_dataset':
            return pd.read_csv('https://raw.githubusercontent.com/datacamp/course-resources-ml-with-experts-budgets/master/data/wine.csv'), ""
        elif asset_name == 'housing_dataset':
            return pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'), ""
        else:
            return None, f"Error loading asset: {str(e)}"


def select_dataset() -> tuple[Optional[pd.DataFrame], str, str]:
    """
    Display a dataset selector widget and return the selected dataset.
    
    Returns:
        Tuple of (DataFrame or None if no selection, asset name, error message if any)
    """
    # Get available assets
    assets = get_available_assets()
    
    # Create a mapping from display name to asset name
    asset_display_names = {
        f"{asset['key']} - {asset['description'][:30] + '...' if len(asset['description']) > 30 else asset['description']}": 
        asset['key'] for asset in assets
    }
    
    # Create dropdown for asset selection
    selected_display_name = st.selectbox(
        "Select Dataset", 
        options=list(asset_display_names.keys()),
        index=0 if asset_display_names else None
    )
    
    if not selected_display_name:
        return None, "", "No dataset selected."
    
    # Get the actual asset name
    selected_asset = asset_display_names[selected_display_name]
    
    # Load the data
    df, error = load_asset_data(selected_asset)
    
    # Display information about the dataset
    if df is not None:
        st.write(f"Dataset: **{selected_asset}**")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df, selected_asset, error


def asset_selector() -> Optional[str]:
    """
    Display a UI for selecting a Dagster asset.
    
    Returns:
        Selected asset name or None if no selection made
    """
    st.markdown("## 📊 Select Data Source")
    
    assets = get_available_dagster_assets()
    
    if not assets:
        st.warning("No Dagster assets found. Please run a Dagster pipeline first.")
        return None
    
    # Add search functionality
    search_term = st.text_input("🔍 Search assets", key="asset_search")
    if search_term:
        filtered_assets = [a for a in assets if search_term.lower() in a.lower()]
    else:
        filtered_assets = assets
    
    if not filtered_assets:
        st.warning(f"No assets matching '{search_term}' found.")
        return None
    
    # Asset selector with categories (if metadata available)
    asset_name = st.selectbox(
        "Select a data asset",
        options=filtered_assets,
        key="asset_selector"
    )
    
    # Show load button
    if st.button("Load Selected Asset", key="load_asset_btn"):
        with st.spinner(f"Loading {asset_name}..."):
            try:
                # Cache the asset name for future use
                st.session_state["selected_asset"] = asset_name
                return asset_name
            except Exception as e:
                st.error(f"Error loading asset: {str(e)}")
                return None
    
    # Return cached selection if already loaded
    return st.session_state.get("selected_asset")


def load_data() -> Optional[pd.DataFrame]:
    """
    Load data from a Dagster asset or a local file.
    
    Returns:
        Optional DataFrame containing the loaded data
    """
    # Initialize session state for data if it doesn't exist
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.data_source = None
        st.session_state.data_path = None
        st.session_state.sample_size = None
        st.session_state.available_assets = None
    
    # Set up UI layout for data selection
    st.write("## Data Source")
    
    # Tabs for different data sources
    data_source_tab = st.radio(
        "Select data source:",
        options=["Dagster Assets", "Local File", "Sample Dataset"],
        horizontal=True
    )
    
    if data_source_tab == "Dagster Assets":
        df = _load_from_dagster()
    elif data_source_tab == "Local File":
        df = _load_from_file()
    else:  # Sample Dataset
        df = _load_sample_dataset()
    
    # Display data preview if data is loaded
    if df is not None:
        st.write("### Data Preview")
        
        # Get the number of rows and columns
        num_rows, num_cols = df.shape
        st.write(f"Dataset shape: {num_rows} rows × {num_cols} columns")
        
        # Sampling controls for large datasets
        if num_rows > 1000:
            sample_size = st.slider(
                "Preview sample size:",
                min_value=100,
                max_value=min(10000, num_rows),
                value=min(1000, num_rows),
                step=100
            )
            st.session_state.sample_size = sample_size
            preview_df = df.sample(sample_size) if sample_size < num_rows else df
        else:
            preview_df = df
            st.session_state.sample_size = num_rows
        
        # Show preview with pagination
        st.dataframe(preview_df, use_container_width=True)
        
        # Show column info
        _display_column_info(df)
        
        # Store the dataframe in session state
        st.session_state.data = df
    
    return st.session_state.data


def _get_dagster_assets() -> list[dict[str, Any]]:
    """
    Get list of available Dagster assets.
    
    Returns:
        List of dictionaries containing asset information
    """
    # Check if we already have the list of assets in session state
    if st.session_state.available_assets is not None:
        return st.session_state.available_assets
    
    try:
        # Try to use the Dagster GraphQL API to get assets
        dagster_url = os.environ.get("DAGSTER_URL", "http://dagster-webserver:3000")
        
        # GraphQL query to get all assets
        query = """
        {
          assetsOrError {
            ... on AssetConnection {
              nodes {
                key {
                  path
                }
                assetMaterializations(limit: 1) {
                  runId
                }
              }
            }
          }
        }
        """
        
        try:
            response = requests.post(
                f"{dagster_url}/graphql",
                json={"query": query},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "assetsOrError" in data["data"]:
                    nodes = data["data"]["assetsOrError"]["nodes"]
                    
                    # Filter to assets that have been materialized
                    assets = [
                        {
                            "name": "/".join(node["key"]["path"]),
                            "path": node["key"]["path"],
                            "materialized": len(node["assetMaterializations"]) > 0
                        }
                        for node in nodes
                        if "key" in node and "path" in node["key"]
                    ]
                    
                    # Sort by materialized status and then by name
                    assets.sort(key=lambda x: (not x["materialized"], x["name"]))
                    
                    st.session_state.available_assets = assets
                    return assets
        except:
            # If GraphQL API fails, fall back to file-based method
            pass
        
        # Fallback approach: Directly examine storage
        # Determine the default dagster home directory
        dagster_home = os.environ.get("DAGSTER_HOME", "/opt/dagster/dagster_home")
        storage_dir = Path(dagster_home) / "storage"
        
        if not storage_dir.exists():
            # Try to find any storage directories in current workspace
            storage_dirs = list(Path.cwd().glob("**/storage"))
            if storage_dirs:
                storage_dir = storage_dirs[0]
            else:
                # If still can't find, default to workspace-relative path
                storage_dir = Path.cwd() / "dagster_home" / "storage"
        
        # Get assets from storage directory
        assets = []
        if storage_dir.exists():
            for asset_dir in storage_dir.glob("**/assets"):
                for asset_path in asset_dir.glob("**/*.parquet"):
                    # Parse the path to get the asset name
                    relative_path = asset_path.relative_to(asset_dir)
                    path_parts = list(relative_path.parts)
                    
                    if path_parts[-1].endswith(".parquet"):
                        # Remove the file extension and any run ID components
                        asset_name = path_parts[-1].split(".")[0]
                        path_parts[-1] = asset_name
                        
                        assets.append({
                            "name": "/".join(path_parts),
                            "path": path_parts,
                            "full_path": str(asset_path),
                            "materialized": True
                        })
        
        # Sort alphabetically
        assets.sort(key=lambda x: x["name"])
        st.session_state.available_assets = assets
        return assets
    
    except Exception as e:
        st.error(f"Error fetching Dagster assets: {e}")
        return []


def _load_from_dagster() -> Optional[pd.DataFrame]:
    """
    Load data from a Dagster asset.
    
    Returns:
        Optional DataFrame containing the loaded data
    """
    st.write("### Select Dagster Asset")
    
    # Get available assets
    assets = _get_dagster_assets()
    
    if not assets:
        st.warning("No Dagster assets found. Check that Dagster is running and assets have been materialized.")
        return None
    
    # Create a selectbox with asset names
    asset_names = [asset["name"] for asset in assets]
    selected_asset = st.selectbox(
        "Available assets:",
        options=asset_names,
        index=0 if asset_names else None
    )
    
    if not selected_asset:
        return None
    
    # Find the selected asset details
    selected_asset_details = next((asset for asset in assets if asset["name"] == selected_asset), None)
    
    if not selected_asset_details:
        st.error("Selected asset not found in available assets.")
        return None
    
    # Load button
    load_clicked = st.button("Load Selected Asset")
    
    if load_clicked:
        with st.spinner(f"Loading asset {selected_asset}..."):
            try:
                # Different loading approaches based on available information
                if "full_path" in selected_asset_details:
                    # Direct file loading if we have the full path
                    df = pd.read_parquet(selected_asset_details["full_path"])
                else:
                    # Try to load using Dagster API
                    asset_path = selected_asset_details["path"]
                    
                    # Attempt to load using Dagster instance
                    dagster_home = os.environ.get("DAGSTER_HOME", "/opt/dagster/dagster_home")
                    instance = DagsterInstance.from_config(dagster_home)
                    
                    # Load the latest materialization
                    asset_key = asset_path
                    latest_materialization = instance.get_latest_materialization_event(asset_key)
                    
                    if latest_materialization:
                        # Get the path from the metadata
                        materialization_path = latest_materialization.event_specific_data.materialization.metadata["path"].value
                        df = pd.read_parquet(materialization_path)
                    else:
                        st.error(f"No materialization found for asset {selected_asset}")
                        return None
                
                # Store data source info
                st.session_state.data_source = "dagster"
                st.session_state.data_path = selected_asset
                
                st.success(f"Successfully loaded asset: {selected_asset}")
                return df
            
            except Exception as e:
                st.error(f"Error loading asset: {str(e)}")
                return None
    
    return None


def _load_from_file() -> Optional[pd.DataFrame]:
    """
    Load data from a local file.
    
    Returns:
        Optional DataFrame containing the loaded data
    """
    st.write("### Upload or Select a File")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls", "parquet"]
    )
    
    # Or browse local files
    local_file = st.text_input(
        "Or enter path to a local file:",
        placeholder="/path/to/your/data.csv"
    )
    
    if uploaded_file is not None:
        # Handle uploaded file
        try:
            with st.spinner("Loading uploaded file..."):
                file_name = uploaded_file.name.lower()
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif file_name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return None
                
                # Store data source info
                st.session_state.data_source = "upload"
                st.session_state.data_path = uploaded_file.name
                
                st.success(f"Successfully loaded file: {uploaded_file.name}")
                return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    elif local_file:
        # Handle local file path
        try:
            with st.spinner(f"Loading file from {local_file}..."):
                if local_file.lower().endswith('.csv'):
                    df = pd.read_csv(local_file)
                elif local_file.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(local_file)
                elif local_file.lower().endswith('.parquet'):
                    df = pd.read_parquet(local_file)
                else:
                    st.error("Unsupported file format")
                    return None
                
                # Store data source info
                st.session_state.data_source = "local"
                st.session_state.data_path = local_file
                
                st.success(f"Successfully loaded file: {local_file}")
                return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    return None


def _load_sample_dataset() -> Optional[pd.DataFrame]:
    """
    Load a sample dataset.
    
    Returns:
        Optional DataFrame containing the loaded data
    """
    st.write("### Select a Sample Dataset")
    
    # List of sample datasets
    sample_datasets = {
        "Iris": "iris",
        "Titanic": "titanic",
        "Diamonds": "diamonds",
        "Housing": "housing",
        "Penguins": "penguins"
    }
    
    selected_dataset = st.selectbox(
        "Sample datasets:",
        options=list(sample_datasets.keys()),
        index=0
    )
    
    load_clicked = st.button("Load Sample Dataset")
    
    if load_clicked:
        with st.spinner(f"Loading sample dataset: {selected_dataset}..."):
            try:
                dataset_key = sample_datasets[selected_dataset]
                
                # Load the selected dataset
                if dataset_key == "iris":
                    from sklearn.datasets import load_iris
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['species'] = pd.Series(data.target).map({
                        0: 'setosa',
                        1: 'versicolor',
                        2: 'virginica'
                    })
                
                elif dataset_key == "titanic":
                    # Try to load from seaborn first, fall back to URL if not available
                    try:
                        import seaborn as sns
                        df = sns.load_dataset('titanic')
                    except:
                        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                        df = pd.read_csv(url)
                
                elif dataset_key == "diamonds":
                    try:
                        import seaborn as sns
                        df = sns.load_dataset('diamonds')
                    except:
                        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
                        df = pd.read_csv(url)
                
                elif dataset_key == "housing":
                    from sklearn.datasets import fetch_california_housing
                    data = fetch_california_housing()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['price'] = data.target
                
                elif dataset_key == "penguins":
                    try:
                        import seaborn as sns
                        df = sns.load_dataset('penguins')
                    except:
                        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
                        df = pd.read_csv(url)
                
                else:
                    st.error(f"Unknown sample dataset: {selected_dataset}")
                    return None
                
                # Store data source info
                st.session_state.data_source = "sample"
                st.session_state.data_path = selected_dataset
                
                st.success(f"Successfully loaded sample dataset: {selected_dataset}")
                return df
            
            except Exception as e:
                st.error(f"Error loading sample dataset: {str(e)}")
                return None
    
    return None


def _display_column_info(df: pd.DataFrame) -> None:
    """
    Display information about DataFrame columns.
    
    Args:
        df: DataFrame to analyze
    """
    st.write("### Column Information")
    
    # Create a dataframe with column information
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        unique_pct = (unique / len(df)) * 100
        
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std()
            }
        else:
            stats = {
                "min": "N/A",
                "max": "N/A", 
                "mean": "N/A",
                "median": "N/A",
                "std": "N/A"
            }
        
        column_info.append({
            "Column": col,
            "Type": dtype,
            "Missing": missing,
            "Missing %": f"{missing_pct:.2f}%",
            "Unique Values": unique,
            "Unique %": f"{unique_pct:.2f}%",
            "Min": stats["min"],
            "Max": stats["max"],
            "Mean": stats["mean"],
            "Median": stats["median"],
            "Std": stats["std"]
        })
    
    # Convert to DataFrame and display
    columns_df = pd.DataFrame(column_info)
    st.dataframe(columns_df, use_container_width=True, hide_index=True) 