"""Fixtures for testing the Streamlit app."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import streamlit as st

# Add root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    # Create mocks for all Streamlit components we need
    streamlit_mocks = {
        "button": MagicMock(return_value=False),
        "file_uploader": MagicMock(return_value=None),
        "text_input": MagicMock(return_value=""),
        "chat_input": MagicMock(return_value=None),
        "form": MagicMock(return_value=MagicMock()),
        "form_submit_button": MagicMock(return_value=False),
        "error": MagicMock(),
        "success": MagicMock(),
        "info": MagicMock(),
        "warning": MagicMock(),
        "subheader": MagicMock(),
        "markdown": MagicMock(),
        "write": MagicMock(),
        "dataframe": MagicMock(),
        "chat_message": MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock())),
        "spinner": MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock())),
        "title": MagicMock(),
        "pyplot": MagicMock(),
        "image": MagicMock(),
        "sidebar": MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock())),
        "rerun": MagicMock(),
        "set_page_config": MagicMock()
    }
    
    # Apply all the patches
    patches = [patch.object(st, name, mock) for name, mock in streamlit_mocks.items()]
    
    # Start all patches
    for p in patches:
        p.start()
    
    # Yield the mocks
    yield streamlit_mocks
    
    # Stop all patches
    for p in patches:
        p.stop()


class SessionStateMock(dict):
    """Mock class for Streamlit's session_state that allows attribute access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getattr__(self, key):
        if key in self:
            return self[key]
        return None
        
    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_session_state():
    """Mock the Streamlit session state with attribute access support."""
    session_state = SessionStateMock()
    with patch.object(st, "session_state", session_state):
        yield session_state


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def mock_agent():
    """Create a mock PandasAI agent."""
    agent = MagicMock()
    agent.chat.return_value = "This is a mock response from PandasAI"
    return agent


@pytest.fixture
def mock_uploaded_file():
    """Create a mock uploaded file."""
    mock_file = MagicMock()
    mock_file.name = "test_data.csv"
    mock_file.getvalue.return_value = b"A,B,C\n1,10,a\n2,20,b\n3,30,c\n4,40,d\n5,50,e"
    return mock_file 