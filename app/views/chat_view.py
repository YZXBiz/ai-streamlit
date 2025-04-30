import os

import streamlit as st


def render_chat_interface(file_name):
    """
    Render the chat interface with input field.

    Args:
        file_name: Name of the file being analyzed

    Returns:
        The user's question if one was submitted, otherwise None
    """
    st.subheader(f"Chat with your data: {file_name}")

    # Display chat history
    display_chat_history()

    # Input for new question
    user_question = st.chat_input("Ask a question about your data...")

    return user_question


def display_chat_history():
    """Display the entire chat history from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "dataframe":
                st.dataframe(message["content"])
            elif message["type"] == "image":
                try:
                    # Display image from file path
                    st.image(message["content"])
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")


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
