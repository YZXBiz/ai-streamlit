"""Tests for the authentication utilities."""

import os
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

from app.utils.auth_utils import (
    get_default_credentials,
    hash_password,
    verify_password,
    authenticate,
    login_form,
    logout
)


def test_get_default_credentials():
    """Test getting default credentials from environment variables."""
    with patch.dict(os.environ, {"DEFAULT_USERNAME": "test_user", "DEFAULT_PASSWORD": "test_pass"}):
        username, password = get_default_credentials()
        assert username == "test_user"
        assert password == "test_pass"

    # Test fallback to defaults
    with patch.dict(os.environ, {}, clear=True):
        username, password = get_default_credentials()
        assert username == "admin"
        assert password == "password"


def test_hash_password():
    """Test password hashing function."""
    with patch.dict(os.environ, {"PASSWORD_SALT": "test_salt"}):
        hashed = hash_password("my_password")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

        # Test consistent hashing
        hashed2 = hash_password("my_password")
        assert hashed == hashed2

        # Different passwords should have different hashes
        hashed3 = hash_password("different_password")
        assert hashed != hashed3


def test_verify_password():
    """Test password verification function."""
    with patch.dict(os.environ, {"PASSWORD_SALT": "test_salt"}):
        password = "my_password"
        hashed = hash_password(password)
        
        # Correct password should verify
        assert verify_password(password, hashed) is True
        
        # Wrong password should not verify
        assert verify_password("wrong_password", hashed) is False


def test_authenticate():
    """Test user authentication function."""
    with patch.dict(os.environ, {"DEFAULT_USERNAME": "test_user", "DEFAULT_PASSWORD": "test_pass"}):
        # Correct credentials should authenticate
        assert authenticate("test_user", "test_pass") is True
        
        # Wrong username should not authenticate
        assert authenticate("wrong_user", "test_pass") is False
        
        # Wrong password should not authenticate
        assert authenticate("test_user", "wrong_pass") is False


def test_login_form(mock_streamlit, mock_session_state):
    """Test the login form functionality."""
    # Test unsuccessful login
    with patch("app.utils.auth_utils.authenticate", return_value=False):
        mock_streamlit["form_submit_button"].return_value = True
        mock_streamlit["text_input"].side_effect = ["test_user", "wrong_pass"]
        
        result = login_form()
        
        assert result is False
        assert mock_streamlit["error"].called
        assert "authenticated" not in mock_session_state

    # Test successful login
    with patch("app.utils.auth_utils.authenticate", return_value=True):
        mock_streamlit["form_submit_button"].return_value = True
        mock_streamlit["text_input"].side_effect = ["test_user", "test_pass"]
        
        result = login_form()
        
        assert result is True
        assert mock_streamlit["success"].called
        assert mock_session_state.get("authenticated") is True


def test_logout(mock_session_state):
    """Test the logout functionality."""
    # Setup session state
    mock_session_state["authenticated"] = True
    mock_session_state["agent"] = "mock_agent"
    mock_session_state["df"] = "mock_df"
    mock_session_state["chat_history"] = ["message1", "message2"]
    
    # Call logout
    result = logout()
    
    # Verify session state was cleared
    assert result is True
    assert mock_session_state["authenticated"] is False
    assert mock_session_state["agent"] is None
    assert mock_session_state["df"] is None
    assert mock_session_state["chat_history"] == [] 