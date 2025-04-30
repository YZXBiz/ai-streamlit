import streamlit as st

from app.controllers.auth_controller import AuthController
from app.controllers.chat_controller import ChatController
from app.controllers.data_controller import DataController
from app.models.session_model import SessionModel
from app.views.layout_view import apply_styling, render_header
from app.views.sidebar_view import render_sidebar


class AppController:
    """Main controller that orchestrates the application flow."""

    def __init__(self) -> None:
        """Initialize the app controller with other controllers."""
        self.session_model = SessionModel()
        self.auth_controller = AuthController()
        self.data_controller = DataController()
        self.chat_controller = ChatController()

    def initialize(self) -> None:
        """Initialize the application state."""
        # Apply custom styling
        apply_styling()

        # Initialize session state
        self.session_model.initialize_session(st.session_state)

        # Render the header
        render_header()

    def run(self) -> None:
        """Run the main application control flow."""
        self.initialize()

        # Authentication check
        if not self.auth_controller.is_authenticated():
            self.auth_controller.handle_login()
            return

        # Render sidebar and get actions
        sidebar_actions = render_sidebar()

        # Handle sidebar actions
        if sidebar_actions["logout"]:
            self.auth_controller.handle_logout()
            st.rerun()

        if sidebar_actions["new_chat"]:
            self.session_model.reset_session(st.session_state)
            st.rerun()

        if sidebar_actions["clear_chat"]:
            self.session_model.reset_chat(st.session_state)
            st.rerun()

        # Main content area based on application state
        if not self.data_controller.has_data():
            # Show file upload interface if no data is loaded
            if self.data_controller.handle_file_upload():
                st.rerun()
        else:
            # Show chat interface if data is loaded
            self.chat_controller.handle_chat()
