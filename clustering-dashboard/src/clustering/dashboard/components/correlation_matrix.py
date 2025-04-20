"""Correlation matrix visualization component."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def create_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] = None,
    colorscale: str = "RdBu_r",
    zmid: float = 0,
    height: int = 600,
    width: int = None,
) -> go.Figure:
    """Generate a correlation matrix visualization for numeric columns.

    Args:
        df: Input DataFrame
        columns: List of column names to include (defaults to all numeric columns)
        colorscale: Plotly colorscale to use for the heatmap
        zmid: Middle value of the colorscale
        height: Height of the figure in pixels
        width: Width of the figure in pixels (if None, uses auto width)

    Returns:
        Plotly figure with correlation matrix
    """
    # If no columns provided, use all numeric columns
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])

    # Check if we have enough columns
    if numeric_df.shape[1] <= 1:
        return None

    # Calculate correlation
    corr = numeric_df.corr()

    # Create heatmap
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=colorscale,
            zmid=zmid,
        )
    )

    # Update layout
    layout_params = {
        "title": "Feature Correlation Matrix",
        "height": height,
    }
    if width:
        layout_params["width"] = width

    fig.update_layout(**layout_params)

    return fig


def display_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] = None,
    with_feature_selector: bool = False,
    max_features: int = 20,
) -> None:
    """Display a correlation matrix with optional feature selector.

    This function combines the correlation matrix visualization with
    an optional UI for selecting which features to include.

    Args:
        df: Input DataFrame
        columns: List of column names to include (defaults to all numeric columns)
        with_feature_selector: Whether to display a feature selector
        max_features: Maximum features to show by default
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) <= 1:
        st.info("Not enough numeric columns for correlation analysis.")
        return

    # Feature selector UI
    if with_feature_selector and len(numeric_cols) > 0:
        selected_features = st.multiselect(
            "Select features to include in correlation matrix:",
            options=numeric_cols,
            default=numeric_cols[: min(max_features, len(numeric_cols))],
        )

        if not selected_features:
            st.warning("Please select at least two features.")
            return

        columns = selected_features

    # Create the correlation matrix
    corr_fig = create_correlation_matrix(df, columns=columns)

    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)

        # Add interpretation and help text
        with st.expander("Understanding the correlation matrix"):
            st.markdown("""
            **Interpreting the correlation matrix:**
            
            - **Values range from -1 to 1**
            - **1**: Perfect positive correlation (as one variable increases, the other increases proportionally)
            - **0**: No correlation (variables are independent)
            - **-1**: Perfect negative correlation (as one variable increases, the other decreases proportionally)
            - **Strong correlations** are generally considered to be above 0.7 or below -0.7
            - **Moderate correlations** are generally between 0.3 and 0.7 (or -0.3 and -0.7)
            - **Weak correlations** are generally below 0.3 (or above -0.3)
            """)
    else:
        st.info("Not enough numeric columns for correlation analysis.")
