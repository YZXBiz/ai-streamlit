"""Main application module for the clustering dashboard.

This is the entry point for the Streamlit dashboard application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import dashboard components
from clustering.dashboard.components.data_loader import (
    load_dagster_asset,
    get_available_assets,
    extract_features,
    extract_cluster_columns,
    create_feature_subset
)
from clustering.dashboard.components.cluster_view import (
    show_cluster_distribution,
    show_feature_scatter,
    show_3d_scatter,
    show_parallel_coordinates,
    show_cluster_comparison
)
from clustering.dashboard.components.feature_explorer import (
    show_feature_distribution,
    show_dimensionality_reduction
)

# Import settings
from clustering.dashboard.config.settings import (
    DASHBOARD_TITLE,
    DASHBOARD_SUBTITLE,
    DEFAULT_ASSETS,
    LAYOUT,
    SIDEBAR_STATE,
    THEME,
    MAX_FEATURES_TO_DISPLAY,
    STORAGE_PATH,
    DATA_PATH,
    CHART_HEIGHT,
    COLOR_SCALES,
    ASSET_DESCRIPTIONS,
    DEFAULT_VISUALIZATIONS
)


def initialize_session_state():
    """Initialize session state variables."""
    if "current_asset" not in st.session_state:
        st.session_state.current_asset = DEFAULT_ASSETS[0]
    
    if "cluster_column" not in st.session_state:
        st.session_state.cluster_column = "final_cluster"
    
    if "compare_data" not in st.session_state:
        st.session_state.compare_data = None
    
    if "compare_column" not in st.session_state:
        st.session_state.compare_column = "merged_cluster"


def setup_page_config():
    """Configure page settings."""
    st.set_page_config(
        page_title=DASHBOARD_TITLE,
        page_icon="📊",
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    
    # Apply custom theme if available
    if THEME:
        # This is a workaround since streamlit.set_theme is not available
        # We'll use custom CSS instead
        css = """
        <style>
        """
        for key, value in THEME.items():
            if key == "primaryColor":
                css += f"""
                a {{color: {value} !important;}}
                .stButton>button {{background-color: {value} !important;}}
                .stProgress .st-bo {{background-color: {value} !important;}}
                """
            elif key == "backgroundColor":
                css += f"""
                .main .block-container {{background-color: {value} !important;}}
                """
            elif key == "secondaryBackgroundColor":
                css += f"""
                .sidebar .sidebar-content {{background-color: {value} !important;}}
                """
            elif key == "textColor":
                css += f"""
                body {{color: {value} !important;}}
                """
            elif key == "font":
                css += f"""
                body {{font-family: {value} !important;}}
                """
        css += """
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


def render_sidebar(available_assets: List[str]):
    """Render the sidebar for asset selection.
    
    Args:
        available_assets: List of available assets
    """
    st.sidebar.title(DASHBOARD_TITLE)
    st.sidebar.caption("A dashboard for exploring cluster assignments")
    
    # Asset selection
    st.sidebar.subheader("Select Data")
    
    # Dropdown for asset selection
    asset_options = []
    for asset in available_assets:
        if asset in ASSET_DESCRIPTIONS:
            asset_options.append(f"{asset} - {ASSET_DESCRIPTIONS[asset]}")
        else:
            asset_options.append(asset)
    
    selected_option = st.sidebar.selectbox(
        "Choose a dataset to analyze",
        options=asset_options,
        index=available_assets.index(st.session_state.current_asset) if st.session_state.current_asset in available_assets else 0
    )
    
    # Extract asset name from the selected option
    selected_asset = selected_option.split(" - ")[0] if " - " in selected_option else selected_option
    
    # Update session state if the asset has changed
    if selected_asset != st.session_state.current_asset:
        st.session_state.current_asset = selected_asset
        # Reset other session state variables that depend on the asset
        st.session_state.compare_data = None
    
    return selected_asset


def load_selected_asset(asset_name: str) -> Tuple[Union[pd.DataFrame, Dict[str, Any], None], str]:
    """Load the selected asset and display loading information.
    
    Args:
        asset_name: Name of the asset to load
        
    Returns:
        Tuple with (loaded_data, path_loaded_from)
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Loading")
    
    with st.sidebar.status("Loading data...") as status:
        data, path = load_dagster_asset(asset_name)
        
        if data is None:
            status.update(label="Error: Data not found", state="error")
            st.error(f"Could not load asset '{asset_name}'. Please check if it exists.")
            return None, ""
        
        status.update(label="Data loaded successfully!", state="complete")
    
    st.sidebar.success(f"Loaded from: {Path(path).name}")
    
    # Display information about the loaded data
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Info")
    
    if isinstance(data, pd.DataFrame):
        st.sidebar.write(f"Type: DataFrame")
        st.sidebar.write(f"Shape: {data.shape}")
        
        # Display sample of the dataframe in an expander
        with st.sidebar.expander("View sample data"):
            st.dataframe(data.head(5), use_container_width=True)
    elif isinstance(data, dict):
        st.sidebar.write(f"Type: Dictionary")
        st.sidebar.write(f"Keys: {list(data.keys())}")
        
        # If it's a dictionary of DataFrames, show info for each
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                with st.sidebar.expander(f"{key} info"):
                    st.write(f"Shape: {value.shape}")
                    st.dataframe(value.head(3), use_container_width=True)
    
    return data, path


def select_analysis_dataframe(data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """Select the DataFrame to analyze and extract relevant columns.
    
    Args:
        data: The loaded data (DataFrame or Dictionary)
        
    Returns:
        Tuple with (selected_dataframe, feature_columns, cluster_columns)
    """
    selected_df = None
    feature_cols = []
    cluster_cols = []
    
    # Different handling based on data type
    if isinstance(data, pd.DataFrame):
        selected_df = data
        feature_cols = extract_features(data)
        cluster_cols = extract_cluster_columns(data)
        
    elif isinstance(data, dict):
        # Select which part of the dictionary to analyze
        dict_keys = list(data.keys())
        
        # Filter to only DataFrames for analysis
        df_keys = [k for k in dict_keys if isinstance(data[k], pd.DataFrame)]
        non_df_keys = [k for k in dict_keys if k not in df_keys]
        
        if df_keys:
            selected_key = st.selectbox(
                "Select data to analyze",
                options=df_keys,
                index=0
            )
            
            selected_df = data[selected_key]
            feature_cols = extract_features(selected_df)
            cluster_cols = extract_cluster_columns(selected_df)
        
        # Display any non-DataFrame elements
        if non_df_keys:
            with st.expander("Additional data elements"):
                for key in non_df_keys:
                    st.write(f"### {key}")
                    st.write(data[key])
    
    return selected_df, feature_cols, cluster_cols


def render_cluster_analysis(df: pd.DataFrame, features: List[str], cluster_cols: List[str]):
    """Render the cluster analysis view.
    
    Args:
        df: DataFrame with cluster assignments
        features: List of feature columns
        cluster_cols: List of cluster columns
    """
    # Only proceed if we have cluster columns
    if not cluster_cols:
        st.warning("No cluster columns found in the data. Cluster columns should contain 'cluster' in their name.")
        return
    
    # Select which cluster column to analyze
    default_index = 0
    if "final_cluster" in cluster_cols:
        default_index = cluster_cols.index("final_cluster")
    elif st.session_state.cluster_column in cluster_cols:
        default_index = cluster_cols.index(st.session_state.cluster_column)
    
    selected_cluster_col = st.selectbox(
        "Select cluster column",
        options=cluster_cols,
        index=default_index
    )
    
    # Update session state
    st.session_state.cluster_column = selected_cluster_col
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cluster Distribution", 
        "Feature Analysis", 
        "Dimension Reduction",
        "Cluster Comparison"
    ])
    
    with tab1:
        st.subheader("Cluster Distribution")
        show_cluster_distribution(df, selected_cluster_col)
    
    with tab2:
        st.subheader("Feature Analysis")
        
        # Limit the number of features shown if there are too many
        if len(features) > MAX_FEATURES_TO_DISPLAY:
            st.info(f"Showing {MAX_FEATURES_TO_DISPLAY} out of {len(features)} features. Use dimension reduction for full analysis.")
            display_features = features[:MAX_FEATURES_TO_DISPLAY]
        else:
            display_features = features
        
        # Create multi-select for features to allow users to select specific ones
        selected_features = st.multiselect(
            "Select features to analyze",
            options=features,
            default=display_features[:min(5, len(display_features))]
        )
        
        if not selected_features:
            st.warning("Please select at least one feature to analyze.")
        else:
            # Feature distribution
            st.write("### Feature Distribution")
            show_feature_distribution(df, selected_features, selected_cluster_col)
            
            # Feature scatter plot
            if len(selected_features) >= 2:
                st.write("### Feature Scatter Plot")
                show_feature_scatter(df, selected_cluster_col, selected_features)
            
            # 3D scatter plot
            if len(selected_features) >= 3:
                st.write("### 3D Scatter Plot")
                show_3d_scatter(df, selected_cluster_col, selected_features)
            
            # Parallel coordinates
            if len(selected_features) >= 3:
                st.write("### Parallel Coordinates")
                show_parallel_coordinates(df, selected_cluster_col, selected_features)
    
    with tab3:
        st.subheader("Dimension Reduction")
        
        if len(features) >= 3:
            show_dimensionality_reduction(df, features, selected_cluster_col)
        else:
            st.warning("Need at least 3 features for dimensionality reduction.")
    
    with tab4:
        st.subheader("Cluster Comparison")
        
        # Check if we have any comparison data
        if st.session_state.compare_data is not None and st.session_state.compare_column:
            show_cluster_comparison(
                st.session_state.compare_data,
                df,
                st.session_state.compare_column,
                selected_cluster_col
            )
        else:
            # Allow loading comparison data
            st.info("To compare clusters, select another dataset to compare with.")
            
            # Let user select which asset to compare with
            available_assets = get_available_assets()
            compare_asset = st.selectbox(
                "Select asset to compare with",
                options=[asset for asset in available_assets if asset != st.session_state.current_asset],
                index=0
            )
            
            if st.button("Load Comparison Data"):
                with st.spinner("Loading comparison data..."):
                    compare_data, _ = load_dagster_asset(compare_asset)
                    
                    if isinstance(compare_data, pd.DataFrame):
                        st.session_state.compare_data = compare_data
                        
                        # Get cluster columns from compare data
                        compare_cluster_cols = extract_cluster_columns(compare_data)
                        
                        if compare_cluster_cols:
                            st.session_state.compare_column = compare_cluster_cols[0]
                            st.success(f"Loaded comparison data with cluster column: {compare_cluster_cols[0]}")
                            st.rerun()
                        else:
                            st.error("No cluster columns found in comparison data.")
                    elif isinstance(compare_data, dict):
                        # Try to find a DataFrame in the dictionary
                        for key, value in compare_data.items():
                            if isinstance(value, pd.DataFrame):
                                st.session_state.compare_data = value
                                
                                # Get cluster columns
                                compare_cluster_cols = extract_cluster_columns(value)
                                
                                if compare_cluster_cols:
                                    st.session_state.compare_column = compare_cluster_cols[0]
                                    st.success(f"Loaded comparison data from key '{key}' with cluster column: {compare_cluster_cols[0]}")
                                    st.rerun()
                                else:
                                    st.error(f"No cluster columns found in comparison data from key '{key}'.")
                                break
                        else:
                            st.error("No suitable DataFrame found in comparison data dictionary.")
                    else:
                        st.error("Comparison data is not in a supported format.")


def main():
    """Main entry point for the dashboard application."""
    # Initialize session state
    initialize_session_state()
    
    # Setup page configuration
    setup_page_config()
    
    # Title and description
    st.title(DASHBOARD_TITLE)
    st.write(DASHBOARD_SUBTITLE)
    
    # Get available assets
    available_assets = get_available_assets()
    
    # Render sidebar and get selected asset
    selected_asset = render_sidebar(available_assets)
    
    # Load selected asset
    data, path = load_selected_asset(selected_asset)
    
    if data is not None:
        # Main content area for data analysis
        st.markdown("---")
        st.header(f"Analysis: {selected_asset}")
        
        # Select which DataFrame to analyze (if data is a dictionary)
        df, features, cluster_cols = select_analysis_dataframe(data)
        
        if df is not None:
            # Render cluster analysis components
            render_cluster_analysis(df, features, cluster_cols)
        else:
            st.warning("No suitable DataFrame found for analysis.")


if __name__ == "__main__":
    main() 