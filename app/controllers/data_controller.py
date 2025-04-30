import streamlit as st

from app.models.agent_model import AgentModel
from app.models.data_model import DataModel
from app.settings import settings
from app.views.uploader_view import render_uploader, show_upload_error, show_upload_success


class DataController:
    """Controller for handling data operations."""

    def __init__(self) -> None:
        """Initialize the data controller with models."""
        self.data_model = DataModel()
        self.agent_model = AgentModel()

    def handle_file_upload(self) -> bool:
        """
        Handle the file upload process.

        Returns:
            True if file was successfully processed, False otherwise
        """
        uploaded_file, continue_clicked = render_uploader()

        if uploaded_file is not None:
            # Process the uploaded file
            df, error = self.data_model.load_dataframe(uploaded_file)

            if error:
                show_upload_error(error)
                return False

            # Get the API key from settings
            api_key = settings.openai_api_key.get_secret_value()
            if not api_key:
                show_upload_error("OpenAI API key not found")
                return False

            # Create the agent
            agent, error = self.agent_model.create_agent(df, api_key)

            if error:
                show_upload_error(f"Error initializing agent: {error}")
                return False

            # Store in session state
            st.session_state.agent = agent
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name

            # Add a welcome message to chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            if not st.session_state.chat_history:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": f"I've analyzed your data from '{uploaded_file.name}'. You can now ask me questions about it!",
                    }
                )

            show_upload_success(uploaded_file.name)

            if continue_clicked:
                return True

        return False

    def has_data(self) -> bool:
        """
        Check if data has been uploaded and agent is initialized.

        Returns:
            True if data is available, False otherwise
        """
        return st.session_state.get("agent") is not None
