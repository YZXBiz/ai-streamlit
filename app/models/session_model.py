from typing import Any


class SessionModel:
    """
    Model for managing the application's session state.
    """

    @staticmethod
    def initialize_session(session: dict[str, Any]) -> None:
        """
        Initialize the session state.

        Args:
            session: The session state dictionary
        """
        if "logged_in" not in session:
            session["logged_in"] = False

        if "chat_history" not in session:
            session["chat_history"] = []

        if "first_question_asked" not in session:
            session["first_question_asked"] = False

        if "agent" not in session:
            session["agent"] = None

        if "dfs" not in session:
            session["dfs"] = {}
            
        if "table_names" not in session:
            session["table_names"] = []
            
        if "file_names" not in session:
            session["file_names"] = []

    @staticmethod
    def reset_session(session: dict[str, Any]) -> None:
        """
        Reset the session state.

        Args:
            session: The session state dictionary
        """
        # Keep login state
        logged_in = session.get("logged_in", False)

        # Clear everything else
        session.clear()

        # Restore login state
        session["logged_in"] = logged_in

        # Re-initialize session
        SessionModel.initialize_session(session)

    @staticmethod
    def reset_chat(session: dict[str, Any]) -> None:
        """
        Reset just the chat history in the session.

        Args:
            session: The session state dictionary
        """
        session["chat_history"] = []
        session["first_question_asked"] = False
