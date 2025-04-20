"""Data Visualization Page for Clustering Dashboard.

This page provides interactive visualization tools for exploring clustering results.
"""

import streamlit as st

from clustering.dashboard.components.parallel_coordinates import (
    display_parallel_coordinates_with_controls,
)
from clustering.dashboard.components.pygwalker_view import get_pyg_renderer
from clustering.dashboard.components.scatter_plot import display_scatter_plot_with_controls


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

        # Use our new scatter plot component with controls
        display_scatter_plot_with_controls(df)

    # Tab: Parallel Coordinates
    with viz_tabs[1]:
        st.subheader("Parallel Coordinates Configuration")

        # Use our new parallel coordinates component with controls
        display_parallel_coordinates_with_controls(df)

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
