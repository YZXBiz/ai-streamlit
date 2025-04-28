"""
Header component for the PandasAI Streamlit application.

This module provides the header component for the main page.
"""

import streamlit as st


def render_header():
    """
    Render the application header.
    
    This function displays the main title and a brief description of the application.
    """
    st.markdown("<h1 class='main-header'>PandasAI Data Assistant 🐼</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='info-box'>
        Ask questions about your data in natural language and get instant insights.
        Upload your data using the sidebar, then ask questions in the chat interface below.
        </div>
        """,
        unsafe_allow_html=True
    )
