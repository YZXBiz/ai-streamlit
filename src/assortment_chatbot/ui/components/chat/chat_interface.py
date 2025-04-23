"""
Component for the Streamlit chat interface in the assortment_chatbot.

This module provides a reusable chat interface component for interacting with retail
assortment data through natural language queries.
"""

from collections.abc import Callable

import streamlit as st

from assortment_chatbot.config.settings import SETTINGS
from assortment_chatbot.core.assortment_analyst import AssortmentAnalyst

# Type for callback function that processes chat messages
MessageProcessor = Callable[[str], str]

# Get settings
agent_settings = SETTINGS.agent_settings
duckdb_settings = SETTINGS.duckdb_settings


def chat_interface(on_message: MessageProcessor) -> None:
    """
    Creates a chat interface component for interacting with assortment data via chat.

    Relies on the `on_message` callback to access necessary data, potentially
    from Streamlit's session state.

    Parameters
    ----------
    on_message : Callable[[str], str]
        Function to call when a message is sent. Takes the message string
        as argument and returns a response string.

    Returns
    -------
    None
        This function modifies the Streamlit UI but doesn't return any values.

    Notes
    -----
    This component requires Streamlit session state to store chat history.
    It initializes `st.session_state.messages` if it doesn't exist.

    Examples
    --------
    >>> def process_message(msg: str) -> str:
    ...     # Access data from session_state if needed
    ...     # df = st.session_state.get('user_data')
    ...     return f"You said: {msg}"
    >>> chat_interface(process_message)
    """
    st.subheader("Assortment Analysis Assistant")

    # Initialize session state variables for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial assistant message if chat is new
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your assortment analysis assistant. Upload your product data to get insights on performance, trends, and optimization opportunities!",
            }
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your product assortment..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Call the callback without the DataFrame
                response = on_message(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a clear button for the chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.assistant = None  # Clear assistant too if desired
        st.rerun()


def display_chat_interface(data=None) -> None:
    """Display the chat interface for interacting with the assortment analyst.

    This component handles:
    - Chat message display
    - User input
    - Message history management
    - Assortment analysis responses

    Args:
        data: Optional data to analyze (typically a pandas DataFrame with product data)
    """
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # If data is passed directly, store it in session state
    if data is not None and not isinstance(data, type(None)):
        st.session_state.user_data = data

    # Debug information about data
    if SETTINGS.DEBUG_MODE and "user_data" in st.session_state:
        st.sidebar.info(
            f"Data info: {type(st.session_state.user_data).__name__} with "
            f"{len(st.session_state.user_data)} rows × {st.session_state.user_data.shape[1]} columns"
        )

    # Initialize assistant if not present
    if "data_assistant" not in st.session_state:
        st.session_state.data_assistant = AssortmentAnalyst()

    # Get assistant instance
    assistant = st.session_state.data_assistant

    # Make sure data is loaded
    assistant.load_data_from_session()

    # Display tables loaded
    available_tables = assistant.get_tables()
    if available_tables:
        st.sidebar.success(f"Assortment data loaded: {', '.join(available_tables)}")
    else:
        st.sidebar.warning("No data loaded. Please upload your assortment data first.")

    # Display settings info if in debug mode
    if SETTINGS.DEBUG_MODE:
        with st.sidebar.expander("Debug: Settings Info", expanded=False):
            st.write(f"Environment: {SETTINGS.ENVIRONMENT}")
            st.write(f"SQL Agent Model: {agent_settings['sql_agent_model']}")
            st.write(f"Interpreter Model: {agent_settings['interpreter_model']}")
            st.write(f"Temperature: {agent_settings['temperature']}")
            st.write(f"DuckDB Path: {duckdb_settings['db_path']}")

    # Chat history
    assistant.get_chat_history()

    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        assistant.clear_chat_history()
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your product assortment..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            try:
                # Show loading spinner while processing
                with st.spinner("Analyzing your assortment..."):
                    # Get response from assistant
                    response = assistant.process_query(prompt)

                # Update placeholder with response
                response_placeholder.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # Show detailed error in debug mode
                if SETTINGS.DEBUG_MODE:
                    import traceback

                    error_msg = (
                        f"Error analyzing query: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
                    )
                else:
                    error_msg = f"Error analyzing query: {str(e)}"

                response_placeholder.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                )

    # Add a clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
