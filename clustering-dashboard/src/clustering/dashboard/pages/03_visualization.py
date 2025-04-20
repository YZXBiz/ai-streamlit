"""Data Visualization Page for Clustering Dashboard.

This page provides interactive visualization tools for exploring clustering results.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering.dashboard.components.pygwalker_view import get_pyg_renderer


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create an interactive scatter plot.

    Args:
        df: DataFrame with the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        color_col: Column to use for coloring points
        size_col: Column to use for point sizes
        title: Plot title

    Returns:
        Plotly figure
    """
    hover_data = list(set([col for col in [x_col, y_col, color_col, size_col] if col]))

    if color_col and size_col:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, size=size_col, hover_data=hover_data, height=600
        )
    elif color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data=hover_data, height=600)
    elif size_col:
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, hover_data=hover_data, height=600)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, hover_data=hover_data, height=600)

    # Set plot title
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f"{y_col} vs. {x_col}")

    # Update layout for better readability
    fig.update_layout(
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def create_parallel_coordinates(
    df: pd.DataFrame, dimension_cols: list[str], color_col: str | None = None
) -> go.Figure:
    """Create a parallel coordinates plot.

    Args:
        df: DataFrame with the data
        dimension_cols: Columns to use as dimensions
        color_col: Column to use for coloring lines

    Returns:
        Plotly figure
    """
    if color_col:
        fig = px.parallel_coordinates(
            df,
            dimensions=dimension_cols,
            color=color_col,
            color_continuous_scale=px.colors.sequential.Viridis,
            height=600,
        )
    else:
        fig = px.parallel_coordinates(
            df,
            dimensions=dimension_cols,
            height=600,
        )

    # Update layout for better readability
    fig.update_layout(
        title="Parallel Coordinates Plot",
        plot_bgcolor="white",
        margin=dict(l=80, r=80, t=60, b=40),
    )

    return fig


def visualization_page():
    """Render the data visualization page."""
    st.title("📊 Data Visualization")

    # Check if data is loaded
    if "data" not in st.session_state:
        st.warning("Please upload data in the Data Upload page first.")
        return

    df = st.session_state["data"]

    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Scatter Plot", "Parallel Coordinates", "Interactive Explorer"])

    # Tab: Scatter Plot
    with viz_tabs[0]:
        st.subheader("Scatter Plot Configuration")

        # Get numeric columns for scatter plot axes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
        else:
            # Layout with two columns
            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols, index=0, key="scatter_x")
                color_col = st.selectbox(
                    "Color by:", [None] + df.columns.tolist(), index=0, key="scatter_color"
                )

            with col2:
                y_col = st.selectbox(
                    "Y-axis:", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y"
                )
                size_col = st.selectbox(
                    "Size by:", [None] + numeric_cols, index=0, key="scatter_size"
                )

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

    # Tab: Parallel Coordinates
    with viz_tabs[1]:
        st.subheader("Parallel Coordinates Configuration")

        # Get numeric columns for dimensions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns for parallel coordinates plot.")
        else:
            # Select dimensions
            dimension_cols = st.multiselect(
                "Select dimensions:",
                options=numeric_cols,
                default=numeric_cols[: min(5, len(numeric_cols))],
                key="pc_dimensions",
            )

            # Select color column
            color_col = st.selectbox("Color by:", [None] + numeric_cols, index=0, key="pc_color")

            if len(dimension_cols) < 2:
                st.warning("Please select at least 2 dimensions.")
            else:
                # Create and display the parallel coordinates plot
                fig = create_parallel_coordinates(
                    df=df,
                    dimension_cols=dimension_cols,
                    color_col=color_col if color_col != "None" else None,
                )

                st.plotly_chart(fig, use_container_width=True)

    # Tab: Interactive Explorer with PyGWalker
    with viz_tabs[2]:
        st.subheader("Interactive Data Explorer")
        st.write("""
        Use this interactive explorer to create various visualizations by dragging and dropping columns.
        You can create scatter plots, bar charts, histograms, and more.
        """)

        # Add PyGWalker renderer
        pyg_html = get_pyg_renderer(df)
        st.components.v1.html(pyg_html, height=1000, scrolling=True)


# Run the page
visualization_page()
