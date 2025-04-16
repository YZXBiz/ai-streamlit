import streamlit as st
import pandas as pd
import polars as pl
import pickle
import os
import numpy as np
import plotly.express as px
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Cluster Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to look for data
possible_data_paths = [
    # From data directory
    "../data/merging/merged_cluster_assignments.pkl",
    # From storage directory
    "../storage/cluster_reassignment",
    "../storage/merged_clusters",
    "../storage/optimized_merged_clusters",
    # Add more potential paths if needed
]

@st.cache_data
def load_data(data_paths):
    """Load cluster data from various possible locations"""
    for path in data_paths:
        if os.path.exists(path):
            st.sidebar.success(f"Found data at: {path}")
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                # Convert Polars DataFrame to pandas if needed
                if hasattr(data, 'to_pandas'):
                    return data.to_pandas(), path
                elif isinstance(data, pl.DataFrame):
                    return data.to_pandas(), path
                elif isinstance(data, pd.DataFrame):
                    return data, path
                elif isinstance(data, dict) and any(isinstance(v, (pd.DataFrame, pl.DataFrame)) for v in data.values()):
                    # Handle dictionary of DataFrames
                    converted_data = {}
                    for k, v in data.items():
                        if hasattr(v, 'to_pandas'):
                            converted_data[k] = v.to_pandas()
                        elif isinstance(v, pl.DataFrame):
                            converted_data[k] = v.to_pandas()
                        else:
                            converted_data[k] = v
                    return converted_data, path
                else:
                    return data, path
            except Exception as e:
                st.sidebar.error(f"Error loading {path}: {e}")
    
    # If no data found, return None
    return None, None

# Header
st.title("Clustering Dashboard")
st.write("Visualize and explore cluster assignments")

# Load data
data, data_path = load_data(possible_data_paths)

if data is None:
    st.error("No data found. Please check data paths.")
    st.stop()

# Show metadata about what was loaded
st.sidebar.subheader("Data Info")
st.sidebar.write(f"Data loaded from: {Path(data_path).name}")

if isinstance(data, dict):
    st.sidebar.write("Data type: Dictionary")
    st.sidebar.write(f"Keys: {list(data.keys())}")
    
    # Let user select which part of the dictionary to visualize
    selected_key = st.sidebar.selectbox("Select data to visualize", options=list(data.keys()))
    
    # If selected item is a DataFrame
    if isinstance(data[selected_key], pd.DataFrame):
        df = data[selected_key]
        st.sidebar.write(f"Shape: {df.shape}")
        st.sidebar.write(f"Columns: {df.columns.tolist()}")
    else:
        st.sidebar.write(f"Selected data type: {type(data[selected_key])}")
        
        # If not a DataFrame, try to display summary
        if isinstance(data[selected_key], dict):
            st.sidebar.write(f"Dictionary keys: {list(data[selected_key].keys())}")
        
        # If it's clusters and small_clusters, show their contents
        if selected_key == "small_clusters" or selected_key == "large_clusters":
            st.write(f"### {selected_key} data")
            st.write(data[selected_key])
else:
    # When data is a DataFrame
    df = data
    st.sidebar.write(f"Data type: {type(df)}")
    st.sidebar.write(f"Shape: {df.shape}")
    st.sidebar.write(f"Columns: {df.columns.tolist()}")

# Visualization options in the main area
st.subheader("Cluster Visualization")

# For DataFrame data
if isinstance(data, pd.DataFrame) or (isinstance(data, dict) and isinstance(data.get(selected_key, None), pd.DataFrame)):
    if isinstance(data, dict):
        df = data[selected_key]
    
    # Get cluster columns and show a sample of the data
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    
    if cluster_cols:
        # Allow selecting which cluster column to visualize
        cluster_col = st.selectbox("Select cluster column", options=cluster_cols, index=len(cluster_cols)-1)
        
        # Show data table with cluster assignments
        st.subheader("Cluster Assignments")
        st.dataframe(df, use_container_width=True)
        
        # Distribution of clusters
        if cluster_col in df.columns:
            st.subheader("Cluster Distribution")
            cluster_counts = df[cluster_col].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            # Bar chart of cluster counts
            fig = px.bar(cluster_counts, x='Cluster', y='Count', 
                          title=f"Distribution of {cluster_col}",
                          labels={'Count': 'Number of Stores', 'Cluster': 'Cluster ID'},
                          color='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        # If there are numeric columns, allow scatter plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != cluster_col and 'store' not in col.lower()]
        
        if len(numeric_cols) >= 2:
            st.subheader("Feature Scatter Plot")
            col1, col2 = st.columns(2)
            x_axis = col1.selectbox("X-axis", options=numeric_cols, index=0)
            y_axis = col2.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
            
            scatter_fig = px.scatter(df, x=x_axis, y=y_axis, color=cluster_col,
                                    title=f"{x_axis} vs {y_axis} by Cluster",
                                    labels={x_axis: x_axis, y_axis: y_axis, cluster_col: 'Cluster'})
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Parallel coordinates for multi-dimensional visualization
        if len(numeric_cols) >= 3:
            st.subheader("Parallel Coordinates Plot")
            
            # Let user select dimensions or use a default set
            selected_dims = st.multiselect(
                "Select dimensions (3-7 recommended)",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_dims) >= 2:
                # Create parallel coordinates plot
                fig = px.parallel_coordinates(
                    df, 
                    dimensions=selected_dims,
                    color=cluster_col,
                    title="Parallel Coordinates Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No cluster columns found in the data. Cluster columns should contain 'cluster' in their name.")
else:
    # For non-DataFrame data or dictionaries without DataFrames
    if isinstance(data, dict):
        st.write("### Data Keys")
        for key in data.keys():
            st.write(f"- {key}: {type(data[key])}")
    else:
        st.write("### Data Summary")
        st.write(f"Type: {type(data)}")
        # If data is a simple type, just display it
        st.write(data)

# Additional controls in sidebar
st.sidebar.subheader("Drag & Drop Features")
st.sidebar.info("Select features to visualize by using the dropdown menus above. For a more interactive experience, consider using streamlit-elements or streamlit-draggable for true drag & drop.")

# Add download button for the data
if isinstance(df, pd.DataFrame):
    st.sidebar.download_button(
        "Download Data as CSV",
        df.to_csv(index=False).encode('utf-8'),
        "cluster_data.csv",
        "text/csv",
        key='download-csv'
    ) 