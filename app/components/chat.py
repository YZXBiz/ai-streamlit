import os

import matplotlib.pyplot as plt
import streamlit as st

from app.utils.pandasai_helper import process_response


def display_chat_history():
    """Display the entire chat history from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "dataframe":
                st.dataframe(message["content"])
            elif message["type"] == "figure":
                # Display matplotlib figure directly
                st.pyplot(message["content"])
            elif message["type"] == "image":
                try:
                    # Display image from file path
                    st.image(message["content"])
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")


def add_message(role, content_type, content):
    """Add a message to the chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": role, "type": content_type, "content": content})


def handle_user_question(question):
    """Process a user question and get response from PandasAI."""
    # Add user question to chat history
    add_message("user", "text", question)

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Check if agent exists
    if "agent" not in st.session_state or st.session_state.agent is None:
        with st.chat_message("assistant"):
            st.error("Please upload a data file first.")
        add_message("assistant", "text", "Please upload a data file first.")
        return

    # Get response from PandasAI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process the question with PandasAI
                response = st.session_state.agent.chat(question)

                # Process the response
                response_type, content = process_response(response)

                # Display the response based on type
                if response_type == "dataframe":
                    st.dataframe(content)
                elif response_type == "figure":
                    # Display matplotlib figure directly
                    st.pyplot(content)
                    # Close figure to prevent memory issues
                    plt.close(content)
                elif response_type == "image":
                    # Check if the file exists
                    if os.path.exists(content):
                        st.image(content)
                    else:
                        st.error(f"Chart file not found: {content}")
                        st.write(str(response))  # Fall back to displaying the raw response
                else:
                    st.write(content)

                # Add response to chat history
                add_message("assistant", response_type, content)

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                add_message("assistant", "text", error_msg)


def chat_interface():
    """Render the chat interface."""
    st.subheader(f"Chat with your data: {st.session_state.get('file_name', 'Dataset')}")

    # Display chat history
    display_chat_history()

    # Input for new question
    user_question = st.chat_input("Ask a question about your data...")

    if user_question:
        handle_user_question(user_question)


def reset_chat():
    """Reset the chat history."""
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
