"""
Authentication utilities for the PandasAI frontend.

This module handles user authentication with the FastAPI backend,
including login, token storage, and authenticated API requests.
"""

import requests
import streamlit as st

# API URL configuration
API_BASE_URL = "http://localhost:8000"  # Update this for production


def login(username: str, password: str) -> dict:
    """
    Authenticate with the backend API using username and password.

    Args:
        username: The user's username
        password: The user's password

    Returns:
        dict: Response from the API containing token or error message
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/login", data={"username": username, "password": password}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def is_authenticated() -> bool:
    """
    Check if the user is currently authenticated.

    Returns:
        bool: True if user has a valid token, False otherwise
    """
    return "token" in st.session_state and st.session_state.token is not None


def get_auth_header() -> dict:
    """
    Get authentication header for API requests.

    Returns:
        dict: Headers containing the authentication token
    """
    if not is_authenticated():
        return {}

    return {"Authorization": f"Bearer {st.session_state.token}"}


def authenticated_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """
    Make an authenticated request to the backend API.

    Args:
        method: HTTP method ('get', 'post', 'put', 'delete')
        endpoint: API endpoint (without base URL)
        **kwargs: Additional arguments to pass to requests

    Returns:
        Response: Response from the API
    """
    url = f"{API_BASE_URL}{endpoint}"

    # Add authentication headers if available
    headers = kwargs.get("headers", {})
    auth_headers = get_auth_header()
    headers.update(auth_headers)
    kwargs["headers"] = headers

    # Make the request with the appropriate method
    request_method = getattr(requests, method.lower())

    try:
        response = request_method(url, **kwargs)

        # Handle unauthorized errors
        if response.status_code == 401:
            st.session_state.token = None
            st.error("Your session has expired. Please log in again.")

        return response
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def logout():
    """
    Log the user out by clearing authentication state.
    """
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.is_authenticated = False

    # Optional: Clear other session state if needed
    if "messages" in st.session_state:
        st.session_state.messages = []
