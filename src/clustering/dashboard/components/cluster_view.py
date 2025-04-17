"""Cluster visualization components for the clustering dashboard.

This module provides visualization components for cluster analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Any, Union

from clustering.dashboard.utils import get_color_scale


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
        color_continuous_scale=get_color_scale("sequential")
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Stores",
        coloraxis_showscale=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        title_font=dict(size=22, color="#FFFFFF"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show counts as a table
    with st.expander("Show cluster counts as table"):
        st.dataframe(cluster_counts)


def show_feature_scatter(df: pd.DataFrame, feature_cols: list[str], 
                        cluster_col: Optional[str] = None):
    """Show scatter plot of features with cluster coloring.
    
    Args:
        df: DataFrame with features and clusters
        feature_cols: Columns containing features
        cluster_col: Column containing cluster labels
    """
    if len(feature_cols) < 2:
        st.warning("Need at least 2 feature columns to create a scatter plot")
        return
        
    st.markdown("### Feature Visualization")
    
    # Chart type selector
    chart_types = {
        "Scatter Plot": "scatter",
        "3D Scatter": "scatter_3d",
        "Bubble Chart": "bubble",
        "Line Plot": "line",
        "Bar Chart": "bar",
        "Box Plot": "box",
        "Violin Plot": "violin",
        "Histogram": "histogram",
        "Density Heatmap": "density_heatmap",
    }
    
    chart_type = st.selectbox(
        "Chart Type",
        options=list(chart_types.keys()),
        index=0
    )
    
    # Create the column selection area with drag-and-drop style
    col1, col2 = st.columns(2)
    
    # Create a feature selector with search functionality
    with col1:
        st.markdown("#### X-Axis")
        
        # Show search box for X axis features
        x_search = st.text_input("Search X-axis features", key="x_search")
        filtered_x_features = [col for col in feature_cols if not x_search or x_search.lower() in col.lower()]
        
        # Ensure we have a default value that exists in the filtered list
        default_x = filtered_x_features[0] if filtered_x_features else feature_cols[0]
        x_feature = st.selectbox(
            "Drag or select feature for X-axis",
            options=filtered_x_features,
            index=filtered_x_features.index(default_x) if default_x in filtered_x_features else 0
        )
    
    with col2:
        st.markdown("#### Y-Axis")
        
        # Show search box for Y axis features
        y_search = st.text_input("Search Y-axis features", key="y_search")
        filtered_y_features = [col for col in feature_cols if not y_search or y_search.lower() in col.lower()]
        
        # Ensure we have a default value that exists in the filtered list
        default_y = filtered_y_features[1] if len(filtered_y_features) > 1 else filtered_y_features[0]
        y_feature = st.selectbox(
            "Drag or select feature for Y-axis",
            options=filtered_y_features,
            index=filtered_y_features.index(default_y) if default_y in filtered_y_features else 0
        )
    
    # For 3D plots, add z-axis
    z_feature = None
    if chart_types[chart_type] == "scatter_3d":
        z_search = st.text_input("Search Z-axis features", key="z_search")
        filtered_z_features = [col for col in feature_cols if not z_search or z_search.lower() in col.lower()]
        default_z = filtered_z_features[2] if len(filtered_z_features) > 2 else filtered_z_features[0]
        z_feature = st.selectbox(
            "Drag or select feature for Z-axis",
            options=filtered_z_features,
            index=filtered_z_features.index(default_z) if default_z in filtered_z_features else 0
        )
    
    # Size variable for bubble charts
    size_var = None
    if chart_types[chart_type] == "bubble":
        size_search = st.text_input("Search size variable", key="size_search")
        filtered_size_features = [col for col in feature_cols if not size_search or size_search.lower() in col.lower()]
        size_var = st.selectbox(
            "Drag or select feature for point size",
            options=filtered_size_features,
            index=0
        )
    
    # Additional visualization options
    with st.expander("Chart Options", expanded=False):
        # Options that make sense for most chart types
        show_trendline = st.checkbox("Show Trendline", value=False)
        trendline_type = None
        if show_trendline and chart_types[chart_type] in ["scatter", "scatter_3d", "line"]:
            trendline_type = st.selectbox(
                "Trendline Type",
                options=["ols", "lowess"],
                index=0
            )
        
        # Color options
        color_scale = st.selectbox(
            "Color Scale",
            options=["Default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", 
                    "Bluered", "RdBu", "Rainbow", "Custom"],
            index=0
        )
        
        # Option to log transform axes
        log_x = st.checkbox("Log scale X-axis", value=False)
        log_y = st.checkbox("Log scale Y-axis", value=False)
        
        # Point customization
        if chart_types[chart_type] in ["scatter", "scatter_3d", "bubble"]:
            point_size = st.slider("Point Size", min_value=2, max_value=15, value=8)
            opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    
    # Plot the data
    try:
        # Get custom color scale if selected
        color_scale_value = None
        if color_scale != "Default":
            if color_scale == "Custom":
                color_scale_value = get_color_scale()
            else:
                color_scale_value = color_scale.lower()
        
        # Start building plot arguments
        plot_args = {
            "x": x_feature,
            "y": y_feature,
            "title": f"{chart_type}: {x_feature} vs {y_feature}",
            "color": cluster_col if cluster_col else None,
            "color_continuous_scale": color_scale_value,
            "labels": {
                x_feature: x_feature.replace("_", " ").title(),
                y_feature: y_feature.replace("_", " ").title(),
            },
        }
        
        # Add z-axis for 3D scatter
        if chart_types[chart_type] == "scatter_3d" and z_feature:
            plot_args["z"] = z_feature
            plot_args["labels"][z_feature] = z_feature.replace("_", " ").title()
            plot_args["title"] = f"3D Scatter: {x_feature}, {y_feature}, {z_feature}"
        
        # Add size for bubble chart
        if chart_types[chart_type] == "bubble" and size_var:
            plot_args["size"] = size_var
            plot_args["size_max"] = 30
        
        # Add trendline if requested
        if trendline_type:
            plot_args["trendline"] = trendline_type
        
        # Add point customization
        if "point_size" in locals():
            if chart_types[chart_type] in ["scatter", "scatter_3d", "bubble"]:
                # For these plot types, marker size is a direct parameter
                plot_args["opacity"] = opacity
                if not size_var:  # Only set marker size if not using a variable for size
                    plot_args["size_max"] = point_size
        
        # Set log axis if requested
        if log_x:
            plot_args["log_x"] = True
        if log_y:
            plot_args["log_y"] = True
        
        # Create the appropriate plot based on chart type
        plot_func = getattr(px, chart_types[chart_type])
        fig = plot_func(df, **plot_args)
        
        # Update layout for better appearance
        fig.update_layout(
            template="plotly_dark" if st.session_state.get("theme") == "dark" else "plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        if chart_types[chart_type] == "scatter_3d":
            st.info("Hint: 3D scatter plots require three numeric columns.")
        elif chart_types[chart_type] in ["histogram", "box", "violin"]:
            st.info(f"Hint: The {chart_type} may need different column types than provided.")


def show_3d_scatter(
    data: pd.DataFrame,
    cluster_col: str,
    features: list[str],
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
    features: list[str],
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