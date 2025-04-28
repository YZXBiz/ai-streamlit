"""
Pytest configuration for frontend tests.

This module contains fixtures and configuration for the frontend tests.
"""

import pytest
from unittest.mock import MagicMock, patch

import streamlit as st
import pandas as pd
import pandasai as pai


@pytest.fixture
def mock_streamlit():
    """Create a mock for Streamlit functions."""
    with patch('streamlit.chat_message') as mock_chat_message, \
         patch('streamlit.chat_input') as mock_chat_input, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.dataframe') as mock_dataframe, \
         patch('streamlit.pyplot') as mock_pyplot, \
         patch('streamlit.code') as mock_code, \
         patch('streamlit.expander') as mock_expander, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.multiselect') as mock_multiselect, \
         patch('streamlit.file_uploader') as mock_file_uploader, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.text_area') as mock_text_area, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.tabs') as mock_tabs:
        
        yield {
            'chat_message': mock_chat_message,
            'chat_input': mock_chat_input,
            'markdown': mock_markdown,
            'dataframe': mock_dataframe,
            'pyplot': mock_pyplot,
            'code': mock_code,
            'expander': mock_expander,
            'spinner': mock_spinner,
            'success': mock_success,
            'error': mock_error,
            'info': mock_info,
            'selectbox': mock_selectbox,
            'multiselect': mock_multiselect,
            'file_uploader': mock_file_uploader,
            'text_input': mock_text_input,
            'text_area': mock_text_area,
            'button': mock_button,
            'sidebar': mock_sidebar,
            'columns': mock_columns,
            'tabs': mock_tabs
        }


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    # Create a mock for st.session_state
    mock_state = {}
    
    # Create a patch for st.session_state that uses our mock
    with patch.object(st, 'session_state', mock_state):
        yield mock_state


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer for testing."""
    mock = MagicMock()
    
    # Mock the dataframe_manager
    mock.dataframe_manager = MagicMock()
    mock.dataframe_manager.get_dataframe.return_value = MagicMock()
    mock.dataframe_manager.get_dataframe_names.return_value = ["test_df"]
    
    # Mock the get_last_code method
    mock.get_last_code.return_value = "print('Hello, world!')"
    
    return mock


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        "country": ["United States", "United Kingdom", "France"],
        "revenue": [5000, 3200, 2900],
        "employees": [150, 90, 80]
    })


@pytest.fixture
def sample_pai_dataframe(sample_dataframe):
    """Create a sample PandasAI DataFrame for testing."""
    return pai.DataFrame(
        sample_dataframe,
        name="test_df",
        description="Test dataframe"
    )
