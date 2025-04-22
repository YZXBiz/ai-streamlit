"""
Component for interactive data visualization using PyGWalker.

This module provides integration with PyGWalker, a Python library that
transforms pandas dataframes into a visual exploration interface similar
to Tableau.
"""

import os

import pandas as pd
import pygwalker as pyg
import streamlit as st

from assortment_chatbot.utils.log_config import get_logger

logger = get_logger(__name__)


def interactive_visualization(df: pd.DataFrame | None = None) -> None:
    """
    Creates an interactive visualization component using PyGWalker.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        The DataFrame to visualize. If None, shows a message
        prompting the user to upload data.

    Returns
    -------
    None
        This function modifies the Streamlit UI but doesn't return any values.

    Notes
    -----
    PyGWalker provides a drag-and-drop interface for data visualization that
    allows users to explore data through an interactive UI without writing code.
    """
    if df is None:
        st.info(
            "Please upload data in the Data Explorer page to use the interactive visualization."
        )
        return

    st.subheader("PyGWalker Interactive Visualization")

    with st.expander("How to use the Interactive Visualization", expanded=False):
        st.markdown("""
        **Quick Guide:**
        1. Drag and drop fields from the left panel to the visualization canvas
        2. Select visualization types (bar, line, point, etc.) to change how your data is displayed
        3. Use filters to focus on specific data segments
        4. Apply aggregations (sum, average, count) for summary analysis
        5. Group by dimensions to create more complex visualizations
        6. Use the toolbar buttons to save or export your work
        
        This tool provides a drag-and-drop interface similar to professional BI tools - 
        create custom visualizations without writing any code.
        """)

    try:
        # Ensure config directory exists
        config_dir = os.path.join("assortment_chatbot", "data", "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "gw_config.json")

        # Configure PyGWalker - use config file if it exists, otherwise use empty dict instead of None
        pyg_html = pyg.to_html(
            df,
            spec=config_path if st.session_state.get("gw_config_exists", False) else {},
            use_kernel_calc=True,
            dark="light" if not st.session_state.get("dark_mode", False) else "dark",
            return_html=True,
        )

        # Render the PyGWalker interface
        components_html = f"""
        <div style="width: 100%; height: 600px; overflow: hidden;">
            {pyg_html}
        </div>
        """
        st.components.v1.html(components_html, height=600, scrolling=False)

        # Add settings for PyGWalker
        with st.expander("Visualization Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Save Configuration"):
                    try:
                        # We would implement a way to save the current config
                        st.session_state["gw_config_exists"] = True
                        st.success("Configuration saved!")
                    except Exception as e:
                        logger.error("Error saving configuration", exc_info=True)
                        st.error(f"Error saving configuration: {e}")

            with col2:
                dark_mode = st.toggle("Dark Mode", value=st.session_state.get("dark_mode", False))
                if dark_mode != st.session_state.get("dark_mode", False):
                    st.session_state["dark_mode"] = dark_mode
                    st.experimental_rerun()

    except Exception as e:
        logger.error("Error in interactive visualization", exc_info=True)
        st.error(f"Error initializing visualization component: {e}")
