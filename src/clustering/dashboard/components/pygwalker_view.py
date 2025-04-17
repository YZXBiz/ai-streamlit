"""PyGWalker integration for the clustering dashboard.

This module provides integration with PyGWalker's visual exploration capabilities
following the latest API recommendations.
"""

import pandas as pd
import streamlit as st


def render_pygwalker(df: pd.DataFrame) -> None:
    """Render the PyGWalker interface for data exploration.

    Args:
        df: DataFrame to visualize
    """
    try:
        # Import the necessary modules
        import pygwalker
        from pygwalker.api.streamlit import StreamlitRenderer

        # Display title and info
        st.markdown("### PyGWalker Visual Explorer")
        st.info(
            "Drag and drop fields to create visualizations. "
            "No coding required! Explore your data visually with this interactive tool."
        )

        # Configure PyGWalker with Streamlit-friendly settings
        config = {
            "hideDataSourceConfig": True,  # Hide data source panel since we're in Streamlit
            "vegaTheme": "g2",  # Use a clean theme
            "dark": "media",  # Adapt to Streamlit's theme
            "enableUserSelection": True,  # Allow users to select data points
        }

        # Create a PyGWalker app with custom config
        pyg_app = StreamlitRenderer(
            df,
            spec_io_mode="rw",  # Allow saving and loading chart configurations
            kernel_computation=True,  # Enable kernel computation for better performance
            theme="media",  # Use media query to match Streamlit's theme
        )

        # Set the height to a fixed value to ensure visibility
        pyg_app.explorer(height=600, config=config)

        # Add a tip for users
        st.caption(
            "Tip: You can export visualizations by clicking the export button in the chart view"
        )

    except ImportError:
        st.error("PyGWalker is not installed. Please install it using: `pip install pygwalker`")

        # Show install instructions
        with st.expander("Installation Instructions"):
            st.code(
                """
                # Install PyGWalker
                pip install pygwalker
                
                # Then restart this application
                """,
                language="bash",
            )

            st.markdown("After installation, restart the Streamlit server to use PyGWalker.")
    except Exception as e:
        st.error(f"Error initializing PyGWalker: {str(e)}")
        if st.session_state.get("show_debugging", False):
            st.exception(e)


def render_pygwalker_with_spec(df: pd.DataFrame, spec: str) -> None:
    """Render PyGWalker with a saved specification.

    Args:
        df: DataFrame to visualize
        spec: PyGWalker chart specification
    """
    try:
        import pygwalker
        from pygwalker.api.streamlit import StreamlitRenderer

        # Display title and info
        st.markdown("### Saved PyGWalker Visualization")

        # Configure PyGWalker with Streamlit-friendly settings
        config = {
            "hideDataSourceConfig": True,  # Hide data source panel since we're in Streamlit
            "vegaTheme": "g2",  # Use a clean theme
            "dark": "media",  # Adapt to Streamlit's theme
            "enableUserSelection": True,  # Allow users to select data points
        }

        # Create a PyGWalker app with the specification
        pyg_app = StreamlitRenderer(
            df,
            spec=spec,
            spec_io_mode="r",  # Read-only for saved specs
            theme="media",  # Use media query to match Streamlit's theme
        )

        # Set the height to a fixed value to ensure visibility
        pyg_app.explorer(height=600, config=config)

    except ImportError:
        st.error("PyGWalker is not installed. Please install it using: `pip install pygwalker`")
    except Exception as e:
        st.error(f"Error initializing PyGWalker with saved specification: {str(e)}")
        if st.session_state.get("show_debugging", False):
            st.exception(e)


@st.cache_resource
def get_pyg_renderer(df: pd.DataFrame, spec: str | None = None) -> object:
    """Get a cached PyGWalker renderer instance.

    This follows the best practice from the PyGWalker docs by using st.cache_resource
    to prevent re-initialization on each rerun.

    Args:
        df: DataFrame to visualize
        spec: Optional chart specification to render

    Returns:
        StreamlitRenderer instance or None if PyGWalker is not installed
    """
    try:
        from pygwalker.api.streamlit import StreamlitRenderer

        # Configure renderer with recommended settings
        renderer = StreamlitRenderer(
            df,
            spec_io_mode="rw",  # Allow saving/loading chart configurations
            dark="media",  # Adapt to Streamlit's theme automatically
            theme="streamlit",  # Use Streamlit-compatible theme
            kernel_computation=True,  # Enable kernel computation for better performance
        )

        return renderer
    except ImportError:
        st.error("PyGWalker is not installed. Please install it using: `pip install pygwalker`")
        return None
