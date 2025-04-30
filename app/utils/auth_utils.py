import hashlib
import hmac
import os

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CredentialManager:
    """Manages user credentials and password hashing."""

    def __init__(self):
        self.salt = os.getenv("PASSWORD_SALT", "default_salt").encode()

    def get_default_credentials(self):
        """Get default username and password from environment variables."""
        default_username = os.getenv("DEFAULT_USERNAME", "admin")
        default_password = os.getenv("DEFAULT_PASSWORD", "password")
        return default_username, default_password

    def hash_password(self, password):
        """Hash a password for storing."""
        return hmac.new(self.salt, password.encode(), hashlib.sha256).hexdigest()

    def verify_password(self, password, hashed_password):
        """Verify a stored password against a provided password."""
        password_hash = self.hash_password(password)
        return password_hash == hashed_password


class Authenticator:
    """Handles user authentication logic."""

    def __init__(self, credential_manager=None):
        self.credential_manager = credential_manager or CredentialManager()

    def authenticate(self, username, password):
        """Authenticate a user with username and password."""
        default_username, default_password = self.credential_manager.get_default_credentials()

        # For development, use simple comparison
        # In production, use hashed passwords
        if username == default_username and password == default_password:
            return True

        return False

    def login_form(self):
        """Display login form and handle authentication."""
        with st.form("login_form"):
            st.title("🔒 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if self.authenticate(username, password):
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    return True
                else:
                    st.error("Invalid username or password")
                    return False

        return False


class SessionManager:
    """Manages user sessions."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def logout(self):
        """Log out the current user."""
        st.session_state.authenticated = False
        # Clear any other session state if needed
        if "agent" in st.session_state:
            st.session_state.agent = None
        if "df" in st.session_state:
            st.session_state.df = None
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []

        return True


# Create singleton instances for use in the application
auth_manager = Authenticator()
session_manager = SessionManager()
