import os

import matplotlib.pyplot as plt
import streamlit as st


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


def add_message(role, content_type, content):
    """Add a message to the chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": role, "type": content_type, "content": content})


def handle_user_question(question):
    """Process a user question and get response from the AI."""
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

    # Get response from AI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process the question with AI
                # First message uses chat(), subsequent messages use follow_up()
                if "first_question_asked" not in st.session_state:
                    response = st.session_state.agent.chat(question)
                    st.session_state.first_question_asked = True
                else:
                    # Use agent.follow_up for subsequent questions
                    response = st.session_state.agent.follow_up(question)

                # Process the response (PandasAI v3 returns objects with attributes)
                response_type = response.type
                
                if response_type == 'chart':
                    # Display image from file path
                    image_path = response.value
                    if os.path.exists(image_path):
                        st.image(image_path)
                        add_message("assistant", "image", image_path)
                    else:
                        st.error(f"Chart file not found: {image_path}")
                        st.write(str(response))
                        add_message("assistant", "text", str(response))
                
                elif response_type == 'string':
                    # Display text response
                    st.write(response.value)
                    add_message("assistant", "text", response.value)
                
                elif response_type == 'dataframe':
                    # Display dataframe
                    df = response.value
                    st.dataframe(df)
                    add_message("assistant", "dataframe", df)
                
                else:
                    # Fallback for other types
                    st.write(response)
                    add_message("assistant", "text", str(response))

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
    if st.button(
        "Clear Chat",
        key="clear_chat_btn",
        help="Clear only the conversation history while keeping the current dataset",
    ):
        st.session_state.chat_history = []
        # Reset first question flag to start a new conversation
        if "first_question_asked" in st.session_state:
            del st.session_state.first_question_asked
        st.rerun()
