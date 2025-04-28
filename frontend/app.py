"""
Main entry point for the PandasAI Streamlit frontend.

This module initializes the Streamlit application and sets up the main page layout.
It serves as a wrapper around the PandasAI backend, providing a conversational
interface for data analysis.
"""

import streamlit as st

from frontend.components.chat_interface import render_chat_interface
from frontend.components.data_preview import render_data_preview
from frontend.components.header import render_header
from frontend.components.login_form import render_login_form, render_login_status
from frontend.components.sidebar import render_sidebar
from frontend.styles.main import apply_custom_styles
from frontend.utils.auth import is_authenticated
from frontend.utils.session import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="PandasAI Data Assistant",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """
    Main function to run the Streamlit application.

    This function:
    1. Applies custom styles
    2. Initializes session state
    3. Checks for authentication
    4. Either displays login form or main application
    5. Renders the sidebar for data management
    6. Renders the main content area with tabs for chat and data preview
    """
    # Apply custom styles
    apply_custom_styles()

    # Initialize session state
    initialize_session_state()

    # Check if user is authenticated
    if not is_authenticated():
        render_login_form()
        return

    # Display login status in sidebar
    render_login_status()

    # Render header
    render_header()

    # Render sidebar for data management
    render_sidebar()

    # Check if dataframes are loaded
    if "loaded_dataframes" not in st.session_state or not st.session_state.loaded_dataframes:
        st.info(
            "👈 Please upload a data file or connect to a database using the sidebar to get started."
        )
        return

    # Create tabs for chat interface and data preview
    tab1, tab2 = st.tabs(["Chat", "Data Preview"])

    # Render chat interface in the first tab
    with tab1:
        render_chat_interface()

    # Render data preview in the second tab
    with tab2:
        render_data_preview()


if __name__ == "__main__":
    main()
