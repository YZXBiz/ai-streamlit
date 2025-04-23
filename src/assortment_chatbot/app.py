"""Main entry point for the assortment_chatbot application."""

from collections.abc import Callable

import streamlit as st
from streamlit_option_menu import option_menu

# Import configuration
from assortment_chatbot.config.constants import (
    APP_DESCRIPTION,
    APP_VERSION,
    CUSTOM_CSS,
    NAV_ITEMS,
    PAGE_CONFIG,
)

# Import UI components
from assortment_chatbot.ui.components.chat.chat_interface import display_chat_interface
from assortment_chatbot.ui.components.data.data_uploader import data_uploader
from assortment_chatbot.ui.components.data.data_viewer import data_viewer
from assortment_chatbot.ui.components.visualization.cluster_viz import cluster_visualization
from assortment_chatbot.ui.components.visualization.pywalker_viz import interactive_visualization

# Setup logging
from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Run the main assortment_chatbot application.

    This function serves as the entry point for the Streamlit application.
    It configures the page and handles the routing to different views based on user navigation.

    Exceptions are caught and displayed to the user.
    """
    try:
        # Initialize basic app settings
        st.set_page_config(**PAGE_CONFIG)

        # Add custom CSS
        add_custom_css()

        # Setup navigation
        selected = setup_navigation()

        # Route to appropriate view
        route_to_view(selected)

    except Exception as e:
        logger.error("Application error", exc_info=True)
        st.error(f"An error occurred: {str(e)}")


def add_custom_css() -> None:
    """Add custom CSS for better styling of the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def setup_navigation() -> str:
    """
    Setup sidebar navigation menu.

    Configures the sidebar navigation using the streamlit_option_menu package
    and displays application information in the sidebar.

    Returns:
        str: The selected navigation item
    """
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            list(NAV_ITEMS.keys()),
            icons=[item["icon"] for item in NAV_ITEMS.values()],
            menu_icon="app-indicator",
            default_index=0,
        )

        # Display app info
        st.sidebar.markdown("---")
        st.sidebar.info(f"Data Chat Assistant {APP_VERSION}\n\n{APP_DESCRIPTION}")

    return selected


def route_to_view(selected: str) -> None:
    """
    Route to the appropriate view based on navigation selection.

    Maps the selected navigation item to the corresponding view handler
    function and executes it.

    Args:
        selected (str): The selected navigation item
    """
    view_handlers: dict[str, Callable[[], None]] = {
        "Home": show_home,
        "Data Uploader": show_data_uploader,
        "Interactive Visualization": show_interactive_visualization,
        "Cluster Analysis": show_cluster_analysis,
        "AI Chat": show_ai_chat,
    }

    view_handler = view_handlers.get(selected)
    if view_handler:
        view_handler()
    else:
        st.error(f"Unknown view: {selected}")


def check_data_loaded() -> bool:
    """
    Check if data is loaded in session state.

    Validates that a user has uploaded data before attempting to
    access data-dependent views.

    Returns:
        bool: True if data is loaded, False otherwise
    """
    if "user_data" not in st.session_state:
        st.warning("Please upload data in the Data Uploader page first.")
        return False
    return True


def show_home() -> None:
    """
    Display the home/welcome page.

    Shows welcome information, features, and getting started guidance
    for the application.
    """
    st.title("🤖 Data Chat Assistant")

    st.markdown(
        f"""
        Welcome to the Data Chat Assistant! This tool helps you explore and 
        understand your data through conversation and visualization.
        
        ## Features
        
        - **Data Explorer**: Upload, preview, and visualize your data
        - **Interactive Visualization**: Create powerful visualizations with drag-and-drop
        - **Cluster Analysis**: Visualize clustering results (if available)
        - **AI Chat**: Chat with your data using natural language queries
        
        ## Getting Started
        
        1. Navigate to the **Data Explorer** page to upload your data
        2. Use the **Interactive Visualization** page to create custom visualizations
        3. Check the **Cluster Analysis** page if you have clustering results
        4. Try the **AI Chat** page to ask questions about your data
        
        ## New in {APP_VERSION}
        
        - **PydanticAI Integration**: Enhanced chat capabilities powered by PydanticAI
        - **SQL Generation**: Ask questions in natural language and get SQL-powered answers
        - **Data Transformation**: Request transformations and download the results
        - **Insightful Analysis**: Get automatic insights about your query results
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


def show_data_uploader() -> None:
    """
    Display the data uploader page with file upload and data preview.

    Handles file uploads and stores the data in session state for use
    in other parts of the application.
    """
    st.title("🔍 Data Uploader")

    # Use the data_uploader component to handle file uploads
    df, filename = data_uploader()

    if df is not None:
        # Store in session state for other pages to use
        st.session_state.user_data = df
        st.session_state.file_name = filename

        # Use the data_viewer component to display and explore the data
        data_viewer(df)


def show_interactive_visualization() -> None:
    """
    Display the PyGWalker interactive visualization page.

    Provides an interactive visualization interface for the uploaded data
    using PyGWalker.
    """
    st.title("📈 Interactive Visualization")

    if not check_data_loaded():
        return

    # Use the interactive_visualization component
    interactive_visualization(st.session_state.user_data)


def show_cluster_analysis() -> None:
    """
    Display cluster analysis if clustering data is available.

    Checks if the uploaded data contains clustering information and
    presents visualization of clusters if available.
    """
    st.title("📈 Cluster Analysis")

    if not check_data_loaded():
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


def show_ai_chat() -> None:
    """
    Display the AI chat interface.

    Presents a chat interface for interacting with the uploaded data
    using natural language queries powered by PydanticAI.
    """
    st.title("💬 Data Chat - Powered by PydanticAI")

    if not check_data_loaded():
        return

    # Get the data from session state
    data = st.session_state.user_data

    # Add a reminder about the data currently being analyzed
    if "file_name" in st.session_state:
        st.info(
            f"Currently analyzing: {st.session_state.file_name} ({len(data)} rows × {data.shape[1]} columns)"
        )
    else:
        st.info(f"Currently analyzing dataset with {len(data)} rows × {data.shape[1]} columns")

    # Display information about the PydanticAI-powered chat
    with st.expander("How to use the Data Chat", expanded=False):
        st.markdown("""
        ## 🔍 How It Works
        
        Ask questions about your data in natural language. The chat assistant will:
        
        1. 🔮 **Generate SQL** to answer your question
        2. 🚀 **Execute the query** using DuckDB
        3. 📊 **Analyze and explain** the results
        4. 💾 **Download transformed data** in various formats
        
        ---
        
        ### 💡 Example Questions
        
        Try asking:
        
        * **Data Exploration:**
          * "Show me the top 5 values in column X"
          * "What's the average of Y grouped by Z?"
        
        * **Data Transformation:**
          * "Create a new column that calculates X divided by Y"
          * "Find outliers in the data"
        
        * **Analysis:**
          * "What's the correlation between A and B?"
          * "Compare the distribution between groups"
          * "What are the key trends in this data?"
        """)

    # Display the chat interface with the uploaded data
    display_chat_interface(data)


if __name__ == "__main__":
    main()
