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

# Direct import of AssortmentAnalyst
from assortment_chatbot.core.assortment_analyst import AssortmentAnalyst

# Import UI components
from assortment_chatbot.ui.components.data.data_uploader import data_uploader
from assortment_chatbot.ui.components.data.data_viewer import data_viewer
from assortment_chatbot.ui.components.home_tab import render_home_tab

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
        "AI Chat": show_ai_chat,
    }

    view_handler = view_handlers.get(selected)
    if view_handler:
        view_handler()
    else:
        st.error(f"Unknown view: {selected}")


def show_home() -> None:
    """
    Display the home/welcome page.

    Shows welcome information, features, and getting started guidance
    for the application.
    """
    render_home_tab()


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
        st.session_state.data_loaded = True

        # Use the data_viewer component to display and explore the data
        data_viewer(df)


def show_ai_chat() -> None:
    """
    Display the AI chat interface.

    Presents a chat interface for interacting with the uploaded data
    using natural language queries powered by the AssortmentAnalyst.
    """
    st.title("🤖 AI Chat Assistant")

    # Check if data is loaded
    if "user_data" not in st.session_state:
        st.warning("Please upload data first in the Data Uploader tab.")

        # Add quick action button to go to data uploader
        if st.button("Go to Data Uploader"):
            st.rerun()
        return

    # Simple chat interface with direct access to AssortmentAnalyst

    # Create AssortmentAnalyst instance directly
    analyst = AssortmentAnalyst()

    # Initialize chat if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load data from session
    analyst.load_data_from_session()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me about your data..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using AssortmentAnalyst directly
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = analyst.process_query(prompt)
                st.markdown(response)

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a clear chat button
    if st.button("Clear Chat") and len(st.session_state.messages) > 0:
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
