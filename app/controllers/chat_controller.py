from typing import Any

import streamlit as st

from app.models.agent_model import AgentModel
from app.views.chat_view import (
    display_assistant_response,
    display_error,
    display_thinking_spinner,
    display_user_message,
    render_chat_interface,
)


class ChatController:
    """Controller for handling chat operations."""

    def __init__(self) -> None:
        """Initialize the chat controller with agent model."""
        self.agent_model = AgentModel()

    def add_message(self, role: str, content_type: str, content: Any) -> None:
        """
        Add a message to the chat history.

        Args:
            role: The role of the message sender ('user' or 'assistant')
            content_type: The type of content ('text', 'dataframe', 'image')
            content: The actual content of the message
        """
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append(
            {"role": role, "type": content_type, "content": content}
        )

    def handle_chat(self, output_type: str = None) -> bool:
        """
        Handle the chat interaction with the agent.

        Args:
            output_type: The desired output type ("auto", "string", "dataframe", "chart").
                         Defaults to None (auto)

        Returns:
            True if a message was processed, False otherwise
        """
        # Get file names and table names from session state
        file_names = st.session_state.get("file_names", [])
        table_names = st.session_state.get("table_names", [])

        # Render chat interface with table information
        user_question = render_chat_interface(file_names, table_names)

        if user_question:
            # Add user message to history and display it
            self.add_message("user", "text", user_question)
            display_user_message(user_question)

            # Get the agent from session state
            agent = st.session_state.get("agent")
            if not agent:
                display_error("Please upload a data file first.")
                self.add_message("assistant", "text", "Please upload a data file first.")
                return False

            # Check if this is the first question
            is_first_question = not st.session_state.get("first_question_asked", False)

            # Process the question with the agent
            with display_thinking_spinner():
                try:
                    if is_first_question:
                        response = agent.chat(user_question, output_type=output_type)
                    else:
                        response = agent.follow_up(user_question, output_type=output_type)

                    # Update the first question flag
                    st.session_state.first_question_asked = True

                    # Display the response
                    display_assistant_response(response)

                    # Add response to chat history
                    response_type = response.type

                    if response_type == "chart":
                        self.add_message("assistant", "image", response.value)
                    elif response_type == "string":
                        self.add_message("assistant", "text", response.value)
                    elif response_type == "dataframe":
                        self.add_message("assistant", "dataframe", response.value)
                    else:
                        self.add_message("assistant", "text", str(response))

                    return True

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    display_error(error_msg)
                    self.add_message("assistant", "text", error_msg)
                    return False

        return False
