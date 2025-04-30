import streamlit as st


def render_login_form():
    """
    Render the login form and get user credentials.

    Returns:
        Tuple of (username, password, submit_clicked)
    """
    with st.form("login_form"):
        st.title("🔒 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        return username, password, submit


def show_login_success():
    """Show a success message after login."""
    st.success("Login successful!")


def show_login_error():
    """Show an error message for failed login."""
    st.error("Invalid username or password")
