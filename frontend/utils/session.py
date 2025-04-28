"""
Session state management for the Streamlit application.

This module provides functions for initializing and managing the Streamlit session state,
which is used to store data between reruns of the app.
"""

import streamlit as st

from backend import create_analyzer


def initialize_session_state():
    """
    Initialize the Streamlit session state with default values.

    This function sets up the following session state variables:
    - analyzer: The PandasAI analyzer instance
    - loaded_dataframes: List of loaded dataframe names
    - messages: Chat message history
    - authentication state: token, is_authenticated, username
    """
    # Initialize the analyzer if it doesn't exist
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = create_analyzer()

    # Initialize the list of loaded dataframes
    if "loaded_dataframes" not in st.session_state:
        st.session_state.loaded_dataframes = []

    # Initialize chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize authentication state
    if "token" not in st.session_state:
        st.session_state.token = None

    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False

    if "username" not in st.session_state:
        st.session_state.username = None


def reset_session_state():
    """
    Reset the session state to its initial values.

    This function is useful for clearing all data and starting fresh.
    """
    # Create a new analyzer instance
    st.session_state.analyzer = create_analyzer()

    # Clear loaded dataframes
    st.session_state.loaded_dataframes = []

    # Clear chat history
    st.session_state.messages = []

    # Preserve authentication state
    # If you want to log out the user, call logout() from auth.py instead
