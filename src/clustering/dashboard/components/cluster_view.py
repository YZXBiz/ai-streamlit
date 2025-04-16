"""Cluster visualization components for the clustering dashboard.

This module provides visualization components for cluster analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional, Tuple, Dict, Any, Union


def show_cluster_distribution(
    data: pd.DataFrame, 
    cluster_col: str,
    title: str = "Cluster Distribution"
) -> None:
    """Display the distribution of stores across clusters.
    
    Args:
        data: DataFrame with cluster assignments
        cluster_col: Name of the cluster column
        title: Title for the visualization
    """
    if cluster_col not in data.columns:
        st.warning(f"Cluster column '{cluster_col}' not found in data")
        return
        
    # Count stores in each cluster
    cluster_counts = data[cluster_col].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    # Sort by cluster name for consistent visualization
    cluster_counts = cluster_counts.sort_values('Cluster')
    
    # Create bar chart
    fig = px.bar(
        cluster_counts, 
        x='Cluster', 
        y='Count',
        title=title,
        labels={'Count': 'Number of Stores', 'Cluster': 'Cluster ID'},
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Stores",
        coloraxis_showscale=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show counts as a table
    with st.expander("Show cluster counts as table"):
        st.dataframe(cluster_counts)


def show_feature_scatter(
    data: pd.DataFrame,
    cluster_col: str,
    features: List[str],
    title_prefix: str = "Cluster Visualization"
) -> None:
    """Show scatter plot of features colored by cluster.
    
    Args:
        data: DataFrame with features and clusters
        cluster_col: Name of the cluster column
        features: List of feature columns to choose from
        title_prefix: Prefix for the plot title
    """
    if len(features) < 2:
        st.warning("Need at least 2 features for scatter plot")
        return
        
    # Let user select features for x and y axes
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", options=features, index=0)
    with col2:
        default_y_index = min(1, len(features) - 1)
        y_axis = st.selectbox("Y-axis", options=features, index=default_y_index)
    
    # Create scatter plot
    fig = px.scatter(
        data, 
        x=x_axis, 
        y=y_axis,
        color=cluster_col,
        title=f"{title_prefix}: {x_axis} vs {y_axis}",
        labels={x_axis: x_axis, y_axis: y_axis, cluster_col: 'Cluster'},
        hover_data=['STORE_NBR'] if 'STORE_NBR' in data.columns else None
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_3d_scatter(
    data: pd.DataFrame,
    cluster_col: str,
    features: List[str],
    title: str = "3D Cluster Visualization"
) -> None:
    """Show 3D scatter plot of features colored by cluster.
    
    Args:
        data: DataFrame with features and clusters
        cluster_col: Name of the cluster column
        features: List of feature columns to choose from
        title: Title for the plot
    """
    if len(features) < 3:
        st.warning("Need at least 3 features for 3D scatter plot")
        return
        
    # Let user select features for axes
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-axis (3D)", options=features, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis (3D)", options=features, index=min(1, len(features) - 1))
    with col3:
        z_axis = st.selectbox("Z-axis", options=features, index=min(2, len(features) - 1))
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        data,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color=cluster_col,
        title=title,
        labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis, cluster_col: 'Cluster'},
        hover_data=['STORE_NBR'] if 'STORE_NBR' in data.columns else None
    )
    
    # Adjust layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_parallel_coordinates(
    data: pd.DataFrame,
    cluster_col: str,
    features: List[str],
    title: str = "Parallel Coordinates Visualization"
) -> None:
    """Show parallel coordinates plot for multi-dimensional visualization.
    
    Args:
        data: DataFrame with features and clusters
        cluster_col: Name of the cluster column
        features: List of available features
        title: Title for the plot
    """
    # Let user select dimensions for visualization
    selected_dims = st.multiselect(
        "Select dimensions (3-7 recommended)",
        options=features,
        default=features[:min(5, len(features))]
    )
    
    if len(selected_dims) < 2:
        st.warning("Please select at least 2 dimensions")
        return
        
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        data,
        dimensions=selected_dims,
        color=cluster_col,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_cluster_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    before_col: str,
    after_col: str,
    title: str = "Cluster Reassignment Comparison"
) -> None:
    """Show a comparison of cluster assignments before and after optimization.
    
    Args:
        before_df: DataFrame with original cluster assignments
        after_df: DataFrame with optimized cluster assignments
        before_col: Column name for original clusters
        after_col: Column name for optimized clusters
        title: Title for the visualization
    """
    if before_col not in before_df.columns or after_col not in after_df.columns:
        st.warning(f"Cluster columns not found in data")
        return
    
    # Merge dataframes if they're not already merged
    if set(before_df.index) != set(after_df.index):
        # Try to merge on STORE_NBR if available
        if 'STORE_NBR' in before_df.columns and 'STORE_NBR' in after_df.columns:
            merged = before_df.merge(after_df, on='STORE_NBR', suffixes=('_before', '_after'))
            before_col = before_col + '_before' if before_col + '_before' in merged.columns else before_col
            after_col = after_col + '_after' if after_col + '_after' in merged.columns else after_col
        else:
            st.warning("Cannot merge before and after dataframes")
            return
    else:
        merged = pd.concat([before_df[before_col], after_df[after_col]], axis=1)
    
    # Count cluster pairs (before -> after)
    cluster_pairs = merged.groupby([before_col, after_col]).size().reset_index()
    cluster_pairs.columns = ['Before', 'After', 'Count']
    
    # Create a Sankey diagram
    source = []  # List of source indices
    target = []  # List of target indices
    value = []   # List of flow values
    
    # Get unique clusters and create mapping to indices
    before_clusters = cluster_pairs['Before'].unique().tolist()
    after_clusters = cluster_pairs['After'].unique().tolist()
    
    # Create mapping from cluster names to indices
    before_to_idx = {cluster: i for i, cluster in enumerate(before_clusters)}
    after_to_idx = {cluster: i + len(before_clusters) for i, cluster in enumerate(after_clusters)}
    
    # Create source, target, value lists
    for _, row in cluster_pairs.iterrows():
        source.append(before_to_idx[row['Before']])
        target.append(after_to_idx[row['After']])
        value.append(row['Count'])
    
    # Create labels
    labels = [f"Before: {c}" for c in before_clusters] + [f"After: {c}" for c in after_clusters]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text=title, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show a summary table
    with st.expander("Show reassignment details"):
        # Count before clusters
        before_counts = merged[before_col].value_counts().reset_index()
        before_counts.columns = ['Cluster', 'Count (Before)']
        
        # Count after clusters
        after_counts = merged[after_col].value_counts().reset_index()
        after_counts.columns = ['Cluster', 'Count (After)']
        
        # Show tables side by side
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Before Optimization")
            st.dataframe(before_counts)
        
        with col2:
            st.write("### After Optimization")
            st.dataframe(after_counts) 