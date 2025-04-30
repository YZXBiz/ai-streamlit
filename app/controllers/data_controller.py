import logging

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
            True if files were successfully processed, False otherwise
        """
        uploaded_files, continue_clicked = render_uploader()

        if uploaded_files and continue_clicked:
            # Process the uploaded files with automatic table naming
            dataframes, errors = self.data_model.load_multiple_dataframes(uploaded_files)

            if errors:
                for error in errors:
                    show_upload_error(error)
                return False

            if not dataframes:
                show_upload_error("No valid data files were uploaded")
                return False

            # Get the API key from settings
            api_key = settings.openai_api_key.get_secret_value()
            if not api_key:
                show_upload_error("OpenAI API key not found")
                return False

            # Log information about the dataframes being processed
            logging.info(f"Creating agent with {len(dataframes)} dataframes")
            for name, df in dataframes.items():
                logging.info(
                    f"Dataframe '{name}' has shape {df.shape} and columns: {', '.join(df.columns)}"
                )

            # Create the agent with multiple dataframes
            agent, error = self.agent_model.create_agent(dataframes, api_key)

            if error:
                show_upload_error(f"Error initializing agent: {error}")
                return False

            # Store in session state
            st.session_state.agent = agent
            st.session_state.dfs = dataframes
            st.session_state.table_names = list(dataframes.keys())

            # Store filenames for display purposes
            file_names = [file.name for file in uploaded_files]
            st.session_state.file_names = file_names

            # Add a welcome message to chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            if not st.session_state.chat_history:
                table_count = len(dataframes)
                tables_str = ", ".join(f"'{name}'" for name in dataframes.keys())

                # Create detailed welcome message showing table info
                welcome_msg = f"I've analyzed your data from {table_count} tables: {tables_str}."

                if table_count > 1:
                    welcome_msg += " You can ask questions that span multiple tables - I'll automatically join related data when needed."
                    # Add hints about table structure
                    welcome_msg += "\n\nData overview:"
                    for name, df in dataframes.items():
                        row_count = len(df)
                        col_count = len(df.columns)
                        welcome_msg += f"\n- '{name}': {row_count} rows, {col_count} columns"

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": welcome_msg,
                    }
                )

            show_upload_success(file_names)
            return True

        return False

    def has_data(self) -> bool:
        """
        Check if data has been uploaded and agent is initialized.

        Returns:
            True if data is available, False otherwise
        """
        return st.session_state.get("agent") is not None
