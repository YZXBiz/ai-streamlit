"""
Login form component for the PandasAI Streamlit application.

This module provides a login interface for authenticating with the backend API.
"""

import streamlit as st

from frontend.utils.auth import login


def render_login_form():
    """
    Render a login form with username and password fields.

    This function displays a form for user authentication and handles
    the login process, storing the token in session state upon success.
    """
    st.title("PandasAI Data Assistant")
    st.subheader("Login")

    # Create login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return

            # Call login function from auth utils
            result = login(username, password)

            if "access_token" in result:
                # Store authentication info in session state
                st.session_state.token = result["access_token"]
                st.session_state.is_authenticated = True
                st.session_state.username = username

                # Show success message
                st.success("Login successful!")

                # Rerun to show main interface
                st.rerun()
            else:
                error_msg = result.get("detail", "Login failed. Please check your credentials.")
                st.error(error_msg)


def render_login_status():
    """
    Render the current login status and logout button if authenticated.

    This function displays the currently logged in user and provides
    a logout button if a user is authenticated.
    """
    if "is_authenticated" in st.session_state and st.session_state.is_authenticated:
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state.username}**")

            if st.button("Logout"):
                # Clear authentication state
                st.session_state.token = None
                st.session_state.is_authenticated = False
                st.session_state.username = None

                # Also clear messages
                if "messages" in st.session_state:
                    st.session_state.messages = []

                # Rerun to show login form
                st.rerun()
