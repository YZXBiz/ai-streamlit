from typing import Any, ClassVar, Optional


class SessionModel:
    """
    Model for managing session state.
    This is a singleton to ensure there's only one instance.
    """

    _instance: ClassVar[Optional["SessionModel"]] = None

    def __new__(cls) -> "SessionModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize_session(self, session_state: dict[str, Any]) -> None:
        """
        Initialize the session state with default values.

        Args:
            session_state: The Streamlit session state
        """
        # Authentication state
        if "authenticated" not in session_state:
            session_state.authenticated = False

        # Data and agent state
        if "agent" not in session_state:
            session_state.agent = None

        if "df" not in session_state:
            session_state.df = None

        if "chat_history" not in session_state:
            session_state.chat_history = []

        if "file_name" not in session_state:
            session_state.file_name = None

        if "first_question_asked" not in session_state:
            session_state.first_question_asked = False

    def logout(self, session_state: dict[str, Any]) -> bool:
        """
        Log out the current user by clearing session state.

        Args:
            session_state: The Streamlit session state

        Returns:
            True if logout is successful
        """
        session_state.authenticated = False

        # Clear other session data
        if "agent" in session_state:
            session_state.agent = None

        if "df" in session_state:
            session_state.df = None

        if "chat_history" in session_state:
            session_state.chat_history = []

        if "file_name" in session_state:
            session_state.file_name = None

        if "first_question_asked" in session_state:
            session_state.first_question_asked = False

        return True

    def reset_chat(self, session_state: dict[str, Any]) -> bool:
        """
        Reset only the chat history while keeping the current dataset.

        Args:
            session_state: The Streamlit session state

        Returns:
            True if reset is successful
        """
        session_state.chat_history = []
        session_state.first_question_asked = False
        return True

    def reset_session(self, session_state: dict[str, Any]) -> bool:
        """
        Reset all session state variables except authentication.

        Args:
            session_state: The Streamlit session state

        Returns:
            True if reset is successful
        """
        session_state.agent = None
        session_state.df = None
        session_state.chat_history = []
        session_state.file_name = None
        session_state.first_question_asked = False
        return True
