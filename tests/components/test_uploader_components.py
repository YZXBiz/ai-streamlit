"""Tests for the file uploader components."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from app.components.uploader_components import file_uploader


def test_file_uploader_no_file(mock_streamlit, mock_session_state):
    """Test file uploader with no file uploaded."""
    # Set file_uploader to return None (no file uploaded)
    mock_streamlit["file_uploader"].return_value = None
    
    # Call the function
    file_uploader()
    
    # Verify expected behaviors
    assert mock_streamlit["subheader"].called
    assert mock_streamlit["info"].called  # Should show info message
    assert mock_streamlit["markdown"].called  # Should display example questions


def test_file_uploader_with_file_error(mock_streamlit, mock_session_state, mock_uploaded_file):
    """Test file uploader with file that causes an error during loading."""
    # Set file_uploader to return a mock file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock load_dataframe to return an error
    with patch('app.components.uploader_components.load_dataframe', return_value=(None, "Test loading error")):
        # Call the function
        file_uploader()
        
        # Verify expected behaviors
        assert mock_streamlit["error"].called
        assert "Test loading error" in mock_streamlit["error"].call_args[0][0]


def test_file_uploader_with_file_no_api_key(mock_streamlit, mock_session_state, mock_uploaded_file, sample_dataframe):
    """Test file uploader with file but no API key."""
    # Set file_uploader to return a mock file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock load_dataframe to return a dataframe
    with patch('app.components.uploader_components.load_dataframe', return_value=(sample_dataframe, None)):
        # First test with empty API key input
        mock_streamlit["text_input"].return_value = ""
        
        # Call the function
        file_uploader()
        
        # Verify expected behaviors
        assert mock_streamlit["warning"].called
        assert "Please enter an OpenAI API key" in mock_streamlit["warning"].call_args[0][0]


def test_file_uploader_with_file_agent_error(mock_streamlit, mock_session_state, mock_uploaded_file, sample_dataframe):
    """Test file uploader with file but agent initialization error."""
    # Set file_uploader to return a mock file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock load_dataframe to return a dataframe
    with patch('app.components.uploader_components.load_dataframe', return_value=(sample_dataframe, None)):
        # Set API key in text input
        mock_streamlit["text_input"].return_value = "test_api_key"
        
        # Mock initialize_agent to return an error
        with patch('app.components.uploader_components.initialize_agent', return_value=(None, "Test agent error")):
            # Call the function
            file_uploader()
            
            # Verify expected behaviors
            assert mock_streamlit["error"].called
            assert "Test agent error" in mock_streamlit["error"].call_args[0][0]


def test_file_uploader_with_file_success(mock_streamlit, mock_session_state, mock_uploaded_file, sample_dataframe, mock_agent):
    """Test file uploader with successful file upload and agent initialization."""
    # Set file_uploader to return a mock file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock load_dataframe to return a dataframe
    with patch('app.components.uploader_components.load_dataframe', return_value=(sample_dataframe, None)):
        # Set API key in session state
        mock_session_state["api_key"] = "existing_api_key"
        
        # Mock initialize_agent to return success
        with patch('app.components.uploader_components.initialize_agent', return_value=(mock_agent, None)), \
             patch('app.components.uploader_components.display_data_info') as mock_display_info:
            # Call the function
            file_uploader()
            
            # Verify expected behaviors
            assert mock_streamlit["success"].called
            assert mock_session_state["agent"] == mock_agent
            assert mock_session_state["df"] is sample_dataframe
            assert mock_session_state["file_name"] == mock_uploaded_file.name
            assert len(mock_session_state["chat_history"]) == 1  # Should add welcome message
            assert mock_display_info.called
            
            # Test the "Continue to Chat" button
            mock_streamlit["button"].return_value = True
            
            with patch('streamlit.rerun') as mock_rerun:
                # Call the function again
                file_uploader()
                
                # Verify expected behaviors
                assert mock_rerun.called


def test_file_uploader_with_api_key_input(mock_streamlit, mock_session_state, mock_uploaded_file, sample_dataframe, mock_agent):
    """Test file uploader with API key input."""
    # Set file_uploader to return a mock file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock load_dataframe to return a dataframe
    with patch('app.components.uploader_components.load_dataframe', return_value=(sample_dataframe, None)):
        # Make sure no API key in session state
        if "api_key" in mock_session_state:
            del mock_session_state["api_key"]
        
        # Set API key input
        mock_streamlit["text_input"].return_value = "input_api_key"
        
        # Mock initialize_agent to return success
        with patch('app.components.uploader_components.initialize_agent', return_value=(mock_agent, None)), \
             patch('app.components.uploader_components.display_data_info'):
            # Call the function
            file_uploader()
            
            # Verify expected behaviors
            assert mock_session_state["api_key"] == "input_api_key" 