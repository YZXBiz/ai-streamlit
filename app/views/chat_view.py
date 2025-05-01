import os

import streamlit as st


def render_chat_interface(file_names=None, table_names=None):
    """
    Render the chat interface with input field.

    Args:
        file_names: List of file names being analyzed
        table_names: List of table names available

    Returns:
        The user's question if one was submitted, otherwise None
    """
    if isinstance(file_names, list) and len(file_names) > 0:
        st.subheader(f"Chat with your data: {', '.join(file_names)}")
    else:
        st.subheader("Chat with your data")

    # Display available tables if more than one
    if table_names and len(table_names) > 1:
        with st.expander("Available Tables"):
            for table in table_names:
                st.write(f"- `{table}`")

            st.write(
                "Tip: You can ask questions that span multiple tables. For best results, reference table names in your questions."
            )

    # Display chat history
    display_chat_history()

    # Input for new question
    if table_names and len(table_names) > 1:
        placeholder = "Ask a question about your data (e.g., 'join the tables and show...')"
    else:
        placeholder = "Ask a question about your data..."

    user_question = st.chat_input(placeholder)

    return user_question


def display_chat_history():
    """Display the entire chat history from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "dataframe":
                st.dataframe(message["content"])
            elif message["type"] == "chart":
                try:
                    # Display image from file path
                    st.image(message["content"])
                except Exception as e:
                    st.error(f"Error displaying chart: {str(e)}")


def display_thinking_spinner():
    """Display a thinking spinner while waiting for AI response."""
    return st.spinner("Thinking...")


def display_user_message(question):
    """Display the user's message in the chat interface."""
    with st.chat_message("user"):
        st.write(question)


def display_assistant_response(response):
    """
    Display the assistant's response in the chat interface.

    Args:
        response: The response object from the agent
    """
    with st.chat_message("assistant"):
        # Process the response (PandasAI v3 returns objects with attributes)
        response_type = response.type

        if response_type == "chart":
            # Display image from file path
            image_path = response.value
            if os.path.exists(image_path):
                st.image(image_path)
            else:
                st.error(f"Chart file not found: {image_path}")
                st.write(str(response))

        elif response_type == "string":
            # Display text response
            st.write(response.value)

        elif response_type == "dataframe":
            # Display dataframe
            df = response.value
            st.dataframe(df)

        else:
            # Fallback for other types
            st.write(response)


def display_error(error_message):
    """Display an error message in the chat interface."""
    with st.chat_message("assistant"):
        st.error(f"Error: {error_message}")
