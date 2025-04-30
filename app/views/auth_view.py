import streamlit as st


def render_login_form():
    """
    Render the login form interface.

    Returns:
        A tuple of (username, password, login_clicked)
    """
    st.title("🔒 Login")
    st.write("Please log in to access the data analysis dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("Login", type="primary", use_container_width=True)

        st.write("*Default login: username 'admin', password 'password'*")

    return username, password, login_button


def show_login_success():
    """Show a success message after login."""
    st.success("Login successful! You can now access the dashboard.")


def show_login_error():
    """Show an error message for failed login."""
    st.error("Invalid username or password. Please try again.")
