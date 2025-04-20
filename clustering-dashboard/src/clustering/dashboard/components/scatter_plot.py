"""Scatter plot visualization component."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    hover_data: list[str] | None = None,
    title: str | None = None,
    height: int = 600,
    width: int = None,
) -> go.Figure:
    """Create an interactive scatter plot.

    Args:
        df: DataFrame with the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        color_col: Column to use for coloring points
        size_col: Column to use for point sizes
        hover_data: List of columns to show in hover tooltip
        title: Plot title
        height: Height of the figure in pixels
        width: Width of the figure in pixels (if None, uses auto width)

    Returns:
        Plotly figure
    """
    # If no hover_data provided, use a reasonable default
    if hover_data is None:
        hover_data = list(set([col for col in [x_col, y_col, color_col, size_col] if col]))

    # Base parameters for the scatter plot
    scatter_params = {
        "data_frame": df,
        "x": x_col,
        "y": y_col,
        "hover_data": hover_data,
        "height": height,
    }

    # Add optional parameters if provided
    if color_col:
        scatter_params["color"] = color_col

    if size_col:
        scatter_params["size"] = size_col

    # Create the scatter plot
    fig = px.scatter(**scatter_params)

    # Set plot title
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f"{y_col} vs. {x_col}")

    # Update layout for better readability
    layout_params = {
        "plot_bgcolor": "white",
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        "margin": dict(l=40, r=40, t=40, b=40),
    }

    if width:
        layout_params["width"] = width

    fig.update_layout(**layout_params)

    return fig


def display_scatter_plot_with_controls(
    df: pd.DataFrame,
    default_x: str | None = None,
    default_y: str | None = None,
    numeric_only: bool = True,
) -> None:
    """Display a scatter plot with interactive controls for customization.

    Args:
        df: DataFrame with the data
        default_x: Default column for x-axis
        default_y: Default column for y-axis
        numeric_only: Whether to only show numeric columns in selectors
    """
    # Get numeric columns if needed
    if numeric_only:
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        available_cols = df.columns.tolist()

    if len(available_cols) < 2:
        st.warning("Need at least 2 columns for scatter plot.")
        return

    # Set default values if not provided
    if default_x is None:
        default_x = available_cols[0]

    if default_y is None:
        default_y = available_cols[min(1, len(available_cols) - 1)]

    # Layout with two columns for controls
    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "X-axis:",
            available_cols,
            index=available_cols.index(default_x) if default_x in available_cols else 0,
            key="scatter_x",
        )

        color_options = [None] + df.columns.tolist()
        color_col = st.selectbox("Color by:", color_options, index=0, key="scatter_color")

    with col2:
        y_col = st.selectbox(
            "Y-axis:",
            available_cols,
            index=available_cols.index(default_y) if default_y in available_cols else 1,
            key="scatter_y",
        )

        size_options = [None] + (
            df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_only
            else df.columns.tolist()
        )
        size_col = st.selectbox("Size by:", size_options, index=0, key="scatter_size")

    # Plot title
    plot_title = st.text_input("Plot Title:", placeholder="Enter a title for your plot")

    # Create and display the scatter plot
    fig = create_scatter_plot(
        df=df,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col if color_col != "None" else None,
        size_col=size_col if size_col != "None" else None,
        title=plot_title if plot_title else None,
    )

    st.plotly_chart(fig, use_container_width=True)
