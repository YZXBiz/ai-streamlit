"""Integration tests for the PandasAI Chat application."""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from app.main import main


@pytest.mark.integration
def test_full_app_workflow(mock_streamlit, mock_session_state, mock_uploaded_file, mock_agent, sample_dataframe):
    """Test the complete application workflow from login to chat."""
    # 1. Start with unauthenticated state
    mock_session_state["authenticated"] = False
    
    # Mock successful login
    with patch('app.utils.auth_utils.authenticate', return_value=True):
        # Setup form submission
        mock_streamlit["form_submit_button"].return_value = True
        mock_streamlit["text_input"].side_effect = ["test_user", "test_pass"]
        
        # Run main once to authenticate
        main()
        
        # Verify user is authenticated
        assert mock_session_state["authenticated"] is True
    
    # 2. Now test file upload and agent initialization
    # Setup file uploader to return a file
    mock_streamlit["file_uploader"].return_value = mock_uploaded_file
    
    # Mock dataframe loading and agent initialization
    with patch('app.components.uploader_components.load_dataframe', return_value=(sample_dataframe, None)), \
         patch('app.components.uploader_components.initialize_agent', return_value=(mock_agent, None)), \
         patch('app.components.uploader_components.display_data_info'):
        
        # Set API key in session state
        mock_session_state["api_key"] = "test_api_key"
        
        # Run main for file upload
        main()
        
        # Verify agent is initialized
        assert mock_session_state["agent"] is mock_agent
        assert mock_session_state["df"] is sample_dataframe
        assert mock_session_state["file_name"] == mock_uploaded_file.name
    
    # 3. Finally, test chat interaction
    # Setup chat input
    mock_streamlit["chat_input"].return_value = "What is the average of column A?"
    
    # Mock response processing
    with patch('app.components.chat_components.process_response', return_value=("text", "The average is 3.0")):
        # Run main for chat
        main()
        
        # Verify chat history has messages
        assert len(mock_session_state["chat_history"]) == 3  # Welcome message + user question + response
        assert mock_session_state["chat_history"][1]["role"] == "user"
        assert mock_session_state["chat_history"][1]["content"] == "What is the average of column A?"
        assert mock_session_state["chat_history"][2]["role"] == "assistant"
        assert mock_session_state["chat_history"][2]["content"] == "The average is 3.0" 