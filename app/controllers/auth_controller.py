import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from app.models.session_model import SessionModel
from app.settings import settings
from app.views.auth_view import render_login_form, show_login_success


class AuthController:
    """Controller for handling authentication."""

    def __init__(self) -> None:
        """Initialize the auth controller with session model and cookie manager."""
        self.session_model = SessionModel()
        self.cookie_manager = self._setup_cookie_manager()

    def _setup_cookie_manager(self) -> EncryptedCookieManager:
        """Initialize and configure the encrypted cookie manager."""
        # Create cookie manager with security key from settings
        cookie_manager = EncryptedCookieManager(
            prefix="chatbot",
            password=settings.cookie_secret,
        )

        # Try to load cookies
        if cookie_manager.ready():
            # Check for existing authentication in cookies
            if cookie_manager.get("user_authenticated") == "true":
                st.session_state.logged_in = True
            else:
                st.session_state.logged_in = False

        return cookie_manager

    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        return st.session_state.get("logged_in", False)

    def handle_login(self) -> bool:
        """
        Handle the login process.

        Returns:
            True if login is successful, False otherwise
        """
        username, password, login_clicked = render_login_form()

        if login_clicked:
            if self._validate_credentials(username, password):
                # Set authenticated flag
                st.session_state.logged_in = True

                # Store authentication in cookies
                if self.cookie_manager.ready():
                    self.cookie_manager["user_authenticated"] = "true"
                    self.cookie_manager.save()

                # Show success message
                show_login_success()

                # Reinitialize session while keeping authenticated state
                self.session_model.initialize_session(st.session_state)

                return True
            else:
                st.error("Invalid credentials")
                return False

        # Not logged in yet
        return False

    def handle_logout(self) -> bool:
        """
        Handle the logout process.

        Returns:
            True if logout is successful
        """
        # Clear cookie on logout
        if self.cookie_manager.ready():
            self.cookie_manager["user_authenticated"] = ""
            self.cookie_manager.save()

        # Reset session and clear user data
        self.session_model.reset_session(st.session_state)

        # Make sure user is logged out
        st.session_state.logged_in = False

        return True

    def _validate_credentials(self, username: str, password: str) -> bool:
        """
        Validate the provided credentials.

        Args:
            username: The username to check
            password: The password to check

        Returns:
            True if credentials are valid, False otherwise
        """
        # Get default credentials from settings
        default_username = settings.default_username
        default_password = settings.default_password

        # Simple credential validation
        return username == default_username and password == default_password
