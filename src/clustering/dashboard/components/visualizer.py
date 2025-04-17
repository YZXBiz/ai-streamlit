import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Optional, Union

def create_visualization(
    df: pd.DataFrame, 
    chart_config: dict[str, Any],
    features: dict[str, Any]
) -> go.Figure:
    """
    Create a visualization based on selected chart type and features.
    
    Args:
        df: DataFrame containing the data
        chart_config: Dictionary with chart type and options
        features: Dictionary with selected features for x, y, color, etc.
    
    Returns:
        Plotly figure object
    """
    chart_type = chart_config["chart_type"]
    options = chart_config.get("options", {})
    
    # Extract features
    x = features.get("x")
    y = features.get("y")
    color = features.get("color")
    size = features.get("size")
    
    # Handle missing features
    if x is None or x not in df.columns:
        st.error(f"X-axis feature '{x}' not found in data.")
        return go.Figure()
    
    if chart_type != "Histogram" and y is None or y not in df.columns:
        st.error(f"Y-axis feature '{y}' not found in data.")
        return go.Figure()
    
    # Basic figure settings
    fig_args = {
        "hover_data": df.columns[:5].tolist()  # Add some default hover data
    }
    
    # Add color if specified
    if color and color in df.columns:
        fig_args["color"] = color
    
    # Add size if specified
    if size and size in df.columns:
        fig_args["size"] = size
    
    # Apply log scales if selected
    if options.get("log_x", False):
        fig_args["log_x"] = True
    
    if options.get("log_y", False):
        fig_args["log_y"] = True
    
    # Create specific chart based on chart type
    if chart_type == "Scatter Plot":
        fig = px.scatter(
            df, x=x, y=y, 
            trendline=options.get("trendline", False) and not color,  # Can't use trendline with color in Plotly Express
            **fig_args
        )
        
    elif chart_type == "3D Scatter":
        # For 3D scatter, we need a third dimension (z)
        # If it wasn't specified, use the first available numeric column other than x and y
        z = features.get("z")
        if not z or z not in df.columns:
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                if col != x and col != y:
                    z = col
                    break
        
        if z:
            fig = px.scatter_3d(
                df, x=x, y=y, z=z,
                **fig_args
            )
        else:
            st.error("No suitable numeric column found for Z-axis in 3D scatter plot.")
            return go.Figure()
            
    elif chart_type == "Bubble Chart":
        if not size or size not in df.columns:
            st.warning("No size feature selected for bubble chart. Using default size.")
            
        fig = px.scatter(
            df, x=x, y=y,
            **fig_args
        )
        
    elif chart_type == "Line Plot":
        fig = px.line(
            df, x=x, y=y,
            markers=options.get("markers", True),
            **fig_args
        )
        
    elif chart_type == "Bar Chart":
        fig = px.bar(
            df, x=x, y=y,
            **fig_args
        )
        
    elif chart_type == "Box Plot":
        fig = px.box(
            df, x=x, y=y,
            **fig_args
        )
        
    elif chart_type == "Violin Plot":
        fig = px.violin(
            df, x=x, y=y,
            box=True,  # Include box plot inside violin
            **fig_args
        )
        
    elif chart_type == "Histogram":
        fig = px.histogram(
            df, x=x,
            nbins=options.get("bins", 20),
            cumulative=options.get("cumulative", False),
            histnorm='probability' if options.get("normalize", False) else None,
            **fig_args
        )
        
    elif chart_type == "Density Heatmap":
        if y:
            fig = px.density_heatmap(
                df, x=x, y=y,
                histnorm='probability' if options.get("normalize", False) else None,
                **fig_args
            )
        else:
            st.error("Y-axis feature required for density heatmap.")
            return go.Figure()
    
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return go.Figure()
    
    # Update layout for better appearance
    fig.update_layout(
        template="plotly_white",
        title=f"{chart_type}: {y if y else ''} vs {x}",
        xaxis_title=x,
        yaxis_title=y if y else "Count",
        legend_title=color if color else "",
        height=600
    )
    
    return fig

def display_visualization(
    df: pd.DataFrame, 
    chart_config: dict[str, Any],
    features: dict[str, Any]
) -> None:
    """
    Display the visualization and related information in the Streamlit app.
    
    Args:
        df: DataFrame containing the data
        chart_config: Dictionary with chart type and options
        features: Dictionary with selected features for x, y, color, etc.
    """
    # Create figure
    fig = create_visualization(df, chart_config, features)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show additional statistical information based on selected features
    with st.expander("Statistical Summary"):
        # Extract features
        x = features.get("x")
        y = features.get("y")
        
        if x and x in df.columns:
            st.write(f"**{x}** Summary Statistics:")
            st.write(df[x].describe())
        
        if y and y in df.columns:
            st.write(f"**{y}** Summary Statistics:")
            st.write(df[y].describe())
        
        # If both x and y are numeric, show correlation
        if x and y and x in df.columns and y in df.columns:
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                correlation = df[x].corr(df[y])
                st.write(f"Correlation between **{x}** and **{y}**: {correlation:.4f}")

def display_data_table(df: pd.DataFrame, features: dict[str, Any]) -> None:
    """
    Display a table of the data with selected features.
    
    Args:
        df: DataFrame containing the data
        features: Dictionary with selected features
    """
    # Determine columns to display
    columns_to_display = []
    
    # Add explicitly selected features
    for feature_type in ["x", "y", "color", "size"]:
        feature = features.get(feature_type)
        if feature and feature in df.columns and feature not in columns_to_display:
            columns_to_display.append(feature)
    
    # Add any additional selected features
    if "selected" in features and features["selected"]:
        for feature in features["selected"]:
            if feature in df.columns and feature not in columns_to_display:
                columns_to_display.append(feature)
    
    # If no columns were selected, show the first few columns
    if not columns_to_display:
        columns_to_display = df.columns[:5].tolist()
    
    # Display the table
    with st.expander("Data Preview"):
        st.dataframe(df[columns_to_display].head(100), use_container_width=True)

def select_chart_type() -> str:
    """
    Allows the user to select a chart type.
    
    Returns:
        The selected chart type
    """
    chart_types = [
        "Scatter Plot",
        "Line Chart",
        "Bar Chart",
        "Histogram",
        "Box Plot",
        "Violin Plot",
        "Heatmap",
        "3D Scatter",
        "Parallel Coordinates",
        "Bubble Chart"
    ]
    
    return st.selectbox("Select chart type", chart_types)

def visualize_data(
    df: pd.DataFrame, 
    selected_features: dict[str, list[str]],
    chart_type: str
) -> Optional[go.Figure]:
    """
    Visualizes the data based on selected features and chart type.
    
    Args:
        df: DataFrame containing the data
        selected_features: Dictionary with selected features for different chart dimensions
        chart_type: The type of chart to create
        
    Returns:
        Plotly figure object or None if visualization couldn't be created
    """
    if df is None or df.empty or not selected_features:
        st.warning("Please select a dataset and features first.")
        return None
    
    # Extract features for each dimension
    x = selected_features.get("x_axis", [None])[0]
    y = selected_features.get("y_axis", [None])[0]
    z = selected_features.get("z_axis", [None])[0]
    color = selected_features.get("color", [None])[0]
    size = selected_features.get("size", [None])[0]
    facet = selected_features.get("facet", [None])[0]
    
    fig = None
    
    try:
        if chart_type == "Scatter Plot":
            if x and y:
                fig = px.scatter(
                    df, x=x, y=y, 
                    color=color, size=size,
                    facet_col=facet,
                    title=f"Scatter Plot: {x} vs {y}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Line Chart":
            if x and y:
                fig = px.line(
                    df, x=x, y=y,
                    color=color,
                    facet_col=facet,
                    title=f"Line Chart: {x} vs {y}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Bar Chart":
            if x and y:
                fig = px.bar(
                    df, x=x, y=y,
                    color=color,
                    facet_col=facet,
                    title=f"Bar Chart: {x} vs {y}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Histogram":
            if x:
                fig = px.histogram(
                    df, x=x,
                    color=color,
                    facet_col=facet,
                    title=f"Histogram of {x}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Box Plot":
            if x and y:
                fig = px.box(
                    df, x=x, y=y,
                    color=color,
                    facet_col=facet,
                    title=f"Box Plot: {x} vs {y}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Violin Plot":
            if x and y:
                fig = px.violin(
                    df, x=x, y=y,
                    color=color,
                    facet_col=facet,
                    title=f"Violin Plot: {x} vs {y}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Heatmap":
            if x and y:
                # Create a pivot table for the heatmap
                pivot_data = df.pivot_table(
                    index=y, columns=x, 
                    values=size if size else df.select_dtypes(include=['number']).columns[0],
                    aggfunc='mean'
                )
                fig = px.imshow(
                    pivot_data,
                    title=f"Heatmap: {x} vs {y}",
                    labels=dict(color="Value")
                )
        
        elif chart_type == "3D Scatter":
            if x and y and z:
                fig = px.scatter_3d(
                    df, x=x, y=y, z=z,
                    color=color, size=size,
                    title=f"3D Scatter: {x} vs {y} vs {z}",
                    hover_data=df.columns
                )
        
        elif chart_type == "Parallel Coordinates":
            # Get all numeric columns for parallel coordinates
            dimensions = list(df.select_dtypes(include=['number']).columns)
            if len(dimensions) > 1:
                fig = px.parallel_coordinates(
                    df, dimensions=dimensions,
                    color=color,
                    title="Parallel Coordinates Plot"
                )
        
        elif chart_type == "Bubble Chart":
            if x and y and size:
                fig = px.scatter(
                    df, x=x, y=y,
                    size=size, color=color,
                    facet_col=facet,
                    title=f"Bubble Chart: {x} vs {y} (size: {size})",
                    hover_data=df.columns
                )
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None
    
    if fig is None:
        st.warning("Could not create visualization with the selected features and chart type.")
        st.write("Please check that you've selected appropriate features for this chart type:")
        if chart_type == "3D Scatter":
            st.write("- 3D Scatter requires X, Y, and Z axes")
        elif chart_type == "Bubble Chart":
            st.write("- Bubble Chart requires X, Y, and Size")
        else:
            st.write(f"- {chart_type} typically requires at least X and Y axes")
    else:
        # Add layout improvements
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
    
    return fig

def configure_chart(fig: go.Figure) -> go.Figure:
    """
    Provides configuration options for the chart.
    
    Args:
        fig: The Plotly figure to configure
        
    Returns:
        The configured figure
    """
    if fig is None:
        return None
    
    st.write("### Chart Configuration")
    
    with st.expander("Chart Options"):
        # Color scheme
        color_scales = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", 
                       "Blues", "Greens", "Reds", "Oranges", "Purples"]
        
        color_scale = st.selectbox("Color scale", color_scales)
        
        # Apply settings
        fig.update_layout(coloraxis=dict(colorscale=color_scale.lower()))
        
        # Chart title
        new_title = st.text_input("Chart title", fig.layout.title.text)
        if new_title:
            fig.update_layout(title=new_title)
        
        # Axes titles
        if hasattr(fig.layout, "xaxis") and fig.layout.xaxis:
            x_title = st.text_input("X-axis title", fig.layout.xaxis.title.text)
            if x_title:
                fig.update_xaxes(title=x_title)
                
        if hasattr(fig.layout, "yaxis") and fig.layout.yaxis:
            y_title = st.text_input("Y-axis title", fig.layout.yaxis.title.text)
            if y_title:
                fig.update_yaxes(title=y_title)
    
    return fig

def download_chart(fig: go.Figure) -> None:
    """
    Provides options to download the chart.
    
    Args:
        fig: The Plotly figure to download
    """
    if fig is None:
        return
    
    with st.expander("Download Chart"):
        col1, col2 = st.columns(2)
        st.dataframe(df[columns_to_display].head(100), use_container_width=True) 