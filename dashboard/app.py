"""Main entry point for the store clustering dashboard application."""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Local imports
from dashboard.llm_assistant import process_chat_message
from dashboard.components.cluster_viz import cluster_visualization
from dashboard.components.data_uploader import data_uploader
from dashboard.components.data_viewer import data_viewer
from dashboard.components.chat_interface import chat_interface

# Set page config
st.set_page_config(
    page_title="Data Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Run the main dashboard application."""
    # Add custom CSS
    add_custom_css()
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Data Explorer", "Cluster Analysis", "AI Chat"],
            icons=["house", "table", "bar-chart", "robot"],
            menu_icon="app-indicator",
            default_index=0,
        )
        
        # Display app info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.info(
            "Data Chat Assistant v0.1.0\n\n"
            "Chat with your data using AI."
        )
    
    # Content based on selection
    if selected == "Home":
        show_home()
    elif selected == "Data Explorer":
        show_data_explorer()
    elif selected == "Cluster Analysis":
        show_cluster_analysis()
    elif selected == "AI Chat":
        show_ai_chat()


def add_custom_css():
    """Add custom CSS for better styling."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stAlert > div {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_home():
    """Display the home/welcome page."""
    st.title("🤖 Data Chat Assistant")
    
    st.markdown(
        """
        Welcome to the Data Chat Assistant! This tool helps you explore and 
        understand your data through conversation and visualization.
        
        ## Features
        
        - **Data Explorer**: Upload, preview, and visualize your data
        - **Cluster Analysis**: Visualize clustering results (if available)
        - **AI Chat**: Chat with your data using natural language queries
        
        ## Getting Started
        
        1. Navigate to the **Data Explorer** page to upload your data
        2. Use the **Cluster Analysis** page if you have clustering results
        3. Try the **AI Chat** page to ask questions about your data
        """
    )
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### Data Upload\nUpload data from CSV, Excel, or JSON files")
    
    with col2:
        st.info("### Interactive Visualizations\nExplore interactive charts and visualizations")
    
    with col3:
        st.info("### AI Assistant\nGet insights through natural language conversation")


def show_data_explorer():
    """Display the data explorer page with file upload and data preview."""
    st.title("📊 Data Explorer")
    
    # Use the data_uploader component to handle file uploads
    df, filename = data_uploader()
    
    if df is not None:
        # Store in session state for other pages to use
        st.session_state.user_data = df
        st.session_state.filename = filename
        
        # Use the data_viewer component to display and explore the data
        data_viewer(df)


def show_cluster_analysis():
    """Display cluster analysis if clustering data is available."""
    st.title("📈 Cluster Analysis")
    
    # Check if data is loaded
    if "user_data" not in st.session_state:
        st.warning("Please upload data in the Data Explorer page first.")
        return
    
    df = st.session_state.user_data
    
    # Check if the data has a CLUSTER or cluster column
    cluster_cols = [col for col in df.columns if col.lower() == "cluster"]
    
    if not cluster_cols:
        st.warning(
            "No clustering information found in the data. "
            "Please upload data that contains a 'CLUSTER' column."
        )
        return
    
    # Use the cluster_visualization component
    cluster_col = cluster_cols[0]
    cluster_visualization(df, cluster_col)


def show_ai_chat():
    """Display the AI chat interface."""
    st.title("💬 AI Chat")
    
    # Check if data is loaded
    if "user_data" not in st.session_state:
        st.warning("Please upload data in the Data Explorer page first.")
        return
    
    df = st.session_state.user_data
    
    # Use the chat_interface component with the process_chat_message function
    chat_interface(process_chat_message, df)


if __name__ == "__main__":
    main() 