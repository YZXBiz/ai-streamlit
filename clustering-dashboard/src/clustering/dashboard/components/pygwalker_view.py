"""Data visualization integration for the clustering dashboard.

This module provides integration with powerful visual exploration capabilities
following modern visualization principles.
"""

import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer


def render_pygwalker(df: pd.DataFrame) -> None:
    """Render the visualization interface for data exploration.

    Args:
        df: DataFrame to visualize
    """

    # Display title and info
    st.markdown("### Visual Data Explorer")
    st.info(
        "Drag and drop fields to create visualizations. "
        "No coding required! Explore your data visually with this interactive tool."
    )

    # Create visualization app with custom config
    viz_app = StreamlitRenderer(
        df,
        spec_io_mode="rw",  # Allow saving and loading chart configurations
        kernel_computation=True,  # Enable kernel computation for better performance
        theme="media",  # Use media query to match Streamlit's theme
    )

    # Call explorer method
    viz_app.explorer()

    # Add a tip for users
    st.caption("Tip: You can export visualizations by clicking the export button in the chart view")


def render_pygwalker_with_spec(df: pd.DataFrame, spec: str) -> None:
    """Render visualization with a saved specification.

    Args:
        df: DataFrame to visualize
        spec: Chart specification
    """
    # Display title and info
    st.markdown("### Saved Visualization")

    # Create visualization app with the specification
    viz_app = StreamlitRenderer(
        df,
        spec=spec,
        spec_io_mode="r",  # Read-only for saved specs
        theme="media",  # Use media query to match Streamlit's theme
    )

    # Call explorer method
    viz_app.explorer()


@st.cache_resource
def get_pyg_renderer(df: pd.DataFrame, spec: str | None = None) -> object:
    """Get a cached visualization renderer instance.

    This uses st.cache_resource to prevent re-initialization on each rerun.

    Args:
        df: DataFrame to visualize
        spec: Optional chart specification to render

    Returns:
        Renderer instance or None if visualization library is not installed
    """

    # Configure renderer with recommended settings
    renderer = StreamlitRenderer(
        df,
        spec_io_mode="rw",  # Allow saving/loading chart configurations
        dark="media",  # Adapt to Streamlit's theme automatically
        theme="streamlit",  # Use Streamlit-compatible theme
        kernel_computation=True,  # Enable kernel computation for better performance
    )

    return renderer


def pygwalker_view(df: pd.DataFrame, title: str = "Data Explorer") -> None:
    """
    Create an interactive data visualization interface using PyGWalker.

    Args:
        df: DataFrame to visualize
        title: Title of the visualization section
    """

    st.header(title)

    # Initialize the renderer
    walker = StreamlitRenderer(df, spec="./gw_config.json", debug=False)
    walker.explorer()
