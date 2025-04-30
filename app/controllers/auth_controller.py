import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from app.models.auth_model import AuthModel
from app.views.auth_view import render_login_form, show_login_error, show_login_success


class AuthController:
    """Controller for handling authentication logic."""

    def __init__(self) -> None:
        """Initialize the auth controller with the auth model."""
        self.auth_model = AuthModel()
        self.cookie_manager = self._setup_cookie_manager()

    def _setup_cookie_manager(self) -> EncryptedCookieManager:
        """Initialize and configure the encrypted cookie manager."""
        from app.settings import settings

        # Create cookie manager with security key from settings
        cookie_manager = EncryptedCookieManager(
            prefix="chatbot",
            password=settings.cookie_secret,
        )

        # Check for existing authentication in cookies
        if cookie_manager.ready():
            if cookie_manager.get("user_authenticated") == "true":
                st.session_state.authenticated = True
            else:
                st.session_state.authenticated = False

        return cookie_manager

    def handle_login(self) -> bool:
        """
        Handle the login process.

        Returns:
            True if login was successful, False otherwise
        """
        username, password, submit = render_login_form()

        if submit:
            if self.auth_model.authenticate(username, password):
                # Update session state
                st.session_state.authenticated = True

                # Set cookie
                if self.cookie_manager.ready():
                    self.cookie_manager["user_authenticated"] = "true"
                    self.cookie_manager.save()

                show_login_success()
                return True
            else:
                show_login_error()

        return False

    def handle_logout(self) -> bool:
        """
        Handle the logout process.

        Returns:
            True if logout was successful
        """
        # Clear cookie on logout
        if self.cookie_manager.ready():
            self.cookie_manager["user_authenticated"] = ""
            self.cookie_manager.save()

        # Update session state
        st.session_state.authenticated = False

        return True

    def is_authenticated(self) -> bool:
        """
        Check if the user is authenticated.

        Returns:
            True if the user is authenticated, False otherwise
        """
        return st.session_state.get("authenticated", False)
