"""
Simplified home tab component for the assortment_chatbot application.

This module provides the landing page for the application.
"""

import logging

import pandas as pd
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)


def render_home_tab() -> None:
    """
    Render the home tab which serves as the landing page for the application.

    Displays an introduction to the simplified interface focusing on the AI Assistant as the main feature.
    """
    st.title("Welcome to Assortment Chatbot")

    # Main intro section
    st.markdown("""
    ## 🤖 Your AI Data Assistant
    
    This simplified application helps you understand and analyze your data through natural conversation.
    
    ### How It Works
    
    1. **Upload your data** - Use the Data Uploader tab to load CSV or Excel files
    2. **Ask questions** - Chat with the AI Assistant in plain English
    3. **Get insights** - View analytical results and download them as CSV
    """)

    st.markdown("---")

    # Quick action buttons
    st.subheader("Get Started")

    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "📊 Import Data",
            on_click=lambda: setattr(st.session_state, "active_tab", "Data Uploader"),
            use_container_width=True,
        )
    with col2:
        st.button(
            "💬 Go to AI Assistant",
            on_click=lambda: setattr(st.session_state, "active_tab", "AI Chat"),
            use_container_width=True,
            type="primary",
            disabled=not st.session_state.get("data_loaded", False),
        )

    # Show current data status
    st.markdown("---")
    st.subheader("Current Data Status")

    if "user_data" in st.session_state:
        data = st.session_state.user_data
        if isinstance(data, pd.DataFrame):
            st.success(f"✅ Data loaded: {data.shape[0]} rows × {data.shape[1]} columns")

            # Show quick data preview
            with st.expander("Data Preview", expanded=False):
                st.dataframe(data.head())
        else:
            st.warning("⚠️ Data format not recognized")
    else:
        st.info("No data loaded yet. Please upload data in the Data Uploader tab.")

    # Example queries
    if "user_data" in st.session_state:
        st.markdown("---")
        st.subheader("Example Questions You Can Ask")
        st.markdown("""
        - Show me the top 5 products by sales
        - What is the average price by category?
        - Which products have inventory below 10 units?
        - Summarize the data by region
        - Plot the sales trend over time
        """)

    # Footer
    st.markdown("---")
    st.caption("Simplified Assortment Chatbot | v0.2.0")
