import streamlit as st
import os
from dotenv import load_dotenv
import hashlib
import hmac

# Load environment variables
load_dotenv()


def get_default_credentials():
    """Get default username and password from environment variables."""
    default_username = os.getenv("DEFAULT_USERNAME", "admin")
    default_password = os.getenv("DEFAULT_PASSWORD", "password")
    return default_username, default_password


def hash_password(password):
    """Hash a password for storing."""
    salt = os.getenv("PASSWORD_SALT", "default_salt").encode()
    return hmac.new(salt, password.encode(), hashlib.sha256).hexdigest()


def verify_password(password, hashed_password):
    """Verify a stored password against a provided password."""
    salt = os.getenv("PASSWORD_SALT", "default_salt").encode()
    password_hash = hmac.new(salt, password.encode(), hashlib.sha256).hexdigest()
    return password_hash == hashed_password


def authenticate(username, password):
    """Authenticate a user with username and password."""
    default_username, default_password = get_default_credentials()
    
    # For development, use simple comparison
    # In production, use hashed passwords
    if username == default_username and password == default_password:
        return True
    
    return False


def login_form():
    """Display login form and handle authentication."""
    with st.form("login_form"):
        st.title("🔒 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                return True
            else:
                st.error("Invalid username or password")
                return False
    
    return False


def logout():
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