"""Parallel coordinates visualization component."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def create_parallel_coordinates(
    df: pd.DataFrame,
    dimension_cols: list[str],
    color_col: str | None = None,
    color_continuous_scale: str = "Viridis",
    height: int = 600,
    width: int = None,
) -> px.parallel_coordinates:
    """Create a parallel coordinates plot.

    Args:
        df: DataFrame with the data
        dimension_cols: Columns to use as dimensions
        color_col: Column to use for coloring lines
        color_continuous_scale: Color scale to use when color_col is provided
        height: Height of the figure in pixels
        width: Width of the figure in pixels (None uses auto-width)

    Returns:
        Plotly parallel coordinates figure
    """
    # Base parameters
    params = {
        "data_frame": df,
        "dimensions": dimension_cols,
        "height": height,
    }

    # Add optional parameters
    if color_col:
        params["color"] = color_col
        params["color_continuous_scale"] = color_continuous_scale

    if width:
        params["width"] = width

    # Create figure
    fig = px.parallel_coordinates(**params)

    # Update layout for better readability
    fig.update_layout(
        title="Parallel Coordinates Plot",
        plot_bgcolor="white",
        margin=dict(l=80, r=80, t=60, b=40),
    )

    return fig


def display_parallel_coordinates_with_controls(
    df: pd.DataFrame,
    max_dimensions: int = 10,
    min_dimensions: int = 2,
) -> None:
    """Display a parallel coordinates plot with interactive controls.

    Args:
        df: DataFrame with the data
        max_dimensions: Maximum number of dimensions to select by default
        min_dimensions: Minimum number of dimensions required
    """
    # Get numeric columns for dimensions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < min_dimensions:
        st.warning(f"Need at least {min_dimensions} numeric columns for parallel coordinates plot.")
        return

    # Select dimensions
    dimension_cols = st.multiselect(
        "Select dimensions:",
        options=numeric_cols,
        default=numeric_cols[: min(max_dimensions, len(numeric_cols))],
        key="pc_dimensions",
    )

    # Select color column
    color_col = st.selectbox(
        "Color by:",
        [None] + numeric_cols,
        index=0,
        key="pc_color",
    )

    # Color scale selector (only shown when color_col is selected)
    color_scale = "Viridis"
    if color_col and color_col != "None":
        color_scale_options = [
            "Viridis",
            "Plasma",
            "Inferno",
            "Magma",
            "Cividis",
            "Blues",
            "Greens",
            "Reds",
            "Oranges",
            "Purples",
            "RdBu",
            "RdBu_r",
            "Spectral",
            "Jet",
        ]
        color_scale = st.selectbox(
            "Color scale:",
            options=color_scale_options,
            index=color_scale_options.index("Viridis"),
            key="pc_color_scale",
        )

    # Warning if not enough dimensions selected
    if len(dimension_cols) < min_dimensions:
        st.warning(f"Please select at least {min_dimensions} dimensions.")
        return

    # Create and display the parallel coordinates plot
    fig = create_parallel_coordinates(
        df=df,
        dimension_cols=dimension_cols,
        color_col=color_col if color_col and color_col != "None" else None,
        color_continuous_scale=color_scale,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanation
    with st.expander("Understanding Parallel Coordinates"):
        st.markdown("""
        **Parallel Coordinates Plot:**
        
        - Each vertical axis represents a different dimension (feature)
        - Each line represents a single data point (row) in your dataset
        - The position where a line crosses an axis represents the value for that dimension
        - Lines with similar paths represent similar data points
        - This visualization is helpful for spotting patterns and clusters in multi-dimensional data
        - When colored by a feature, it helps identify how that feature relates to others
        """)
