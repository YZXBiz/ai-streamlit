"""
Tests for the main frontend application.

This module contains tests for the main Streamlit application.
"""

import pytest
from unittest.mock import patch, MagicMock

import streamlit as st
from frontend.app import main


@pytest.fixture
def mock_streamlit():
    """Create mocks for Streamlit functions."""
    with patch('frontend.app.st') as mock_st:
        # Mock the set_page_config function
        mock_st.set_page_config = MagicMock()
        
        # Mock the tabs function
        mock_tabs = (MagicMock(), MagicMock())
        mock_st.tabs.return_value = mock_tabs
        
        # Mock the info function
        mock_st.info = MagicMock()
        
        yield mock_st


@pytest.fixture
def mock_components():
    """Create mocks for the component functions."""
    with patch('frontend.app.apply_custom_styles') as mock_styles, \
         patch('frontend.app.initialize_session_state') as mock_init, \
         patch('frontend.app.render_sidebar') as mock_sidebar, \
         patch('frontend.app.render_header') as mock_header, \
         patch('frontend.app.render_chat_interface') as mock_chat, \
         patch('frontend.app.render_data_preview') as mock_preview:
        
        yield {
            'styles': mock_styles,
            'init': mock_init,
            'sidebar': mock_sidebar,
            'header': mock_header,
            'chat': mock_chat,
            'preview': mock_preview
        }


def test_main_no_dataframes(mock_streamlit, mock_components):
    """Test the main function when no dataframes are loaded."""
    # Set up the session state
    with patch.object(st, 'session_state', {}):
        # Call the main function
        main()
        
        # Check that the components were called correctly
        mock_components['styles'].assert_called_once()
        mock_components['init'].assert_called_once()
        mock_components['sidebar'].assert_called_once()
        mock_components['header'].assert_called_once()
        
        # Check that the info message was displayed
        mock_streamlit.info.assert_called_once()
        
        # Check that the chat and preview components were not called
        mock_components['chat'].assert_not_called()
        mock_components['preview'].assert_not_called()


def test_main_with_dataframes(mock_streamlit, mock_components):
    """Test the main function when dataframes are loaded."""
    # Set up the session state with loaded dataframes
    with patch.object(st, 'session_state', {'loaded_dataframes': ['test_df']}):
        # Call the main function
        main()
        
        # Check that the components were called correctly
        mock_components['styles'].assert_called_once()
        mock_components['init'].assert_called_once()
        mock_components['sidebar'].assert_called_once()
        mock_components['header'].assert_called_once()
        
        # Check that the tabs were created
        mock_streamlit.tabs.assert_called_once_with(["Chat", "Data Preview"])
        
        # Check that the chat and preview components were called
        mock_components['chat'].assert_called_once()
        mock_components['preview'].assert_called_once()
        
        # Check that the info message was not displayed
        mock_streamlit.info.assert_not_called()
