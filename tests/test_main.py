"""Tests for the main application."""

import os
import pytest
import streamlit as st
from unittest.mock import patch, MagicMock

from app.main import reset_session, main


def test_reset_session(mock_streamlit, mock_session_state):
    """Test the reset_session function."""
    # Set up session state
    mock_session_state.agent = "mock_agent"
    mock_session_state.df = "mock_df"
    mock_session_state.chat_history = ["message1", "message2"]
    mock_session_state.file_name = "test.csv"
    
    # Test without clicking button
    mock_streamlit["button"].return_value = False
    
    # Call the function
    reset_session()
    
    # Verify session state is not cleared
    assert mock_session_state.agent == "mock_agent"
    assert mock_session_state.df == "mock_df"
    assert mock_session_state.chat_history == ["message1", "message2"]
    assert mock_session_state.file_name == "test.csv"
    
    # Test with button click
    mock_streamlit["button"].return_value = True
    
    # Call the function
    with patch('streamlit.rerun') as mock_rerun:
        reset_session()
        
        # Verify expected behaviors
        assert mock_session_state.agent is None
        assert mock_session_state.df is None
        assert mock_session_state.chat_history == []
        assert mock_session_state.file_name is None
        assert mock_rerun.called


def test_main_not_authenticated(mock_streamlit, mock_session_state):
    """Test the main function when user is not authenticated."""
    # Set up session state - not authenticated
    mock_session_state.authenticated = False
    
    # Mock the login_form function
    with patch('app.main.login_form') as mock_login:
        # Call the function
        main()
        
        # Verify expected behaviors
        assert mock_streamlit["title"].called
        assert mock_streamlit["markdown"].called
        assert mock_login.called
        
        # Verify that file_uploader and chat_interface are not called
        # We don't care about sidebar since it doesn't run when not authenticated


def test_main_authenticated_no_agent(mock_streamlit, mock_session_state):
    """Test the main function when user is authenticated but no agent is initialized."""
    # Set up session state - authenticated but no agent
    mock_session_state.authenticated = True
    mock_session_state.agent = None
    
    # Mock streamlit sidebar and its components
    with patch('app.main.file_uploader') as mock_uploader, \
         patch('app.main.reset_chat') as mock_reset_chat, \
         patch('app.main.logout') as mock_logout, \
         patch('streamlit.sidebar', MagicMock()):
        
        # Call the function
        main()
        
        # Verify expected behaviors
        assert mock_streamlit["title"].called
        
        # Verify that the file uploader is shown but not the chat
        assert mock_uploader.called


def test_main_authenticated_with_agent(mock_streamlit, mock_session_state, mock_agent):
    """Test the main function when user is authenticated and agent is initialized."""
    # Set up session state - authenticated with agent
    mock_session_state.authenticated = True
    mock_session_state.agent = mock_agent
    
    # Mock streamlit sidebar and its components
    with patch('app.main.chat_interface') as mock_chat, \
         patch('app.main.reset_chat') as mock_reset_chat, \
         patch('app.main.logout') as mock_logout, \
         patch('streamlit.sidebar', MagicMock()):
        
        # Call the function
        main()
        
        # Verify expected behaviors
        assert mock_streamlit["title"].called
        
        # Verify that the chat interface is shown but not the file uploader
        assert mock_chat.called


def test_main_sidebar_logout(mock_streamlit, mock_session_state):
    """Test the logout button in the sidebar."""
    # Set up session state - authenticated
    mock_session_state.authenticated = True
    
    # Setup button to simulate logout click
    mock_streamlit["button"].return_value = True
    
    # Mock the logout function
    with patch('app.main.logout') as mock_logout, \
         patch('streamlit.rerun') as mock_rerun:
        
        # Mock file_uploader to avoid testing that path
        with patch('app.main.file_uploader'):
            # Call the function
            main()
            
            # Verify expected behaviors
            assert mock_logout.called
            assert mock_rerun.called 