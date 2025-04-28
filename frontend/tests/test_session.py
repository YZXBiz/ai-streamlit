"""
Tests for the session state utility.

This module contains tests for the session state management functions.
"""

import pytest
import streamlit as st
from unittest.mock import MagicMock, patch

from frontend.utils.session import initialize_session_state, reset_session_state


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
    return MagicMock()


def test_initialize_session_state(mock_session_state, mock_analyzer):
    """Test that initialize_session_state correctly sets up the session state."""
    # Patch the create_analyzer function to return our mock
    with patch('frontend.utils.session.create_analyzer', return_value=mock_analyzer):
        # Call the function
        initialize_session_state()
        
        # Check that the session state was initialized correctly
        assert 'analyzer' in mock_session_state
        assert 'loaded_dataframes' in mock_session_state
        assert 'messages' in mock_session_state
        
        assert mock_session_state['loaded_dataframes'] == []
        assert mock_session_state['messages'] == []


def test_reset_session_state(mock_session_state, mock_analyzer):
    """Test that reset_session_state correctly resets the session state."""
    # Set up the session state with some data
    mock_session_state['analyzer'] = "old_analyzer"
    mock_session_state['loaded_dataframes'] = ["df1", "df2"]
    mock_session_state['messages'] = [{"role": "user", "content": "test"}]
    
    # Patch the create_analyzer function to return our mock
    with patch('frontend.utils.session.create_analyzer', return_value=mock_analyzer):
        # Call the function
        reset_session_state()
        
        # Check that the session state was reset correctly
        assert mock_session_state['analyzer'] == mock_analyzer
        assert mock_session_state['loaded_dataframes'] == []
        assert mock_session_state['messages'] == []
