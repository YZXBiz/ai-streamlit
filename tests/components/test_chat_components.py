"""Tests for the chat components."""

import os
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import pandas as pd

from app.components.chat_components import (
    display_chat_history,
    add_message,
    handle_user_question,
    chat_interface,
    reset_chat
)


def test_add_message(mock_session_state):
    """Test adding a message to the chat history."""
    # Test adding first message
    add_message("user", "text", "Hello")
    
    assert hasattr(mock_session_state, "chat_history")
    assert len(mock_session_state.chat_history) == 1
    assert mock_session_state.chat_history[0]["role"] == "user"
    assert mock_session_state.chat_history[0]["type"] == "text"
    assert mock_session_state.chat_history[0]["content"] == "Hello"
    
    # Test adding second message
    add_message("assistant", "text", "Hi there!")
    
    assert len(mock_session_state.chat_history) == 2
    assert mock_session_state.chat_history[1]["role"] == "assistant"
    assert mock_session_state.chat_history[1]["type"] == "text"
    assert mock_session_state.chat_history[1]["content"] == "Hi there!"


def test_display_chat_history(mock_streamlit, mock_session_state):
    """Test displaying chat history."""
    # Setup mock session state with different message types
    mock_session_state.chat_history = [
        {"role": "user", "type": "text", "content": "Show me data"},
        {"role": "assistant", "type": "dataframe", "content": pd.DataFrame({"A": [1, 2], "B": [3, 4]})},
        {"role": "user", "type": "text", "content": "Show me a chart"},
        {"role": "assistant", "type": "figure", "content": plt.figure()},
        {"role": "user", "type": "text", "content": "Show me an image"},
        {"role": "assistant", "type": "image", "content": "path/to/image.png"}
    ]
    
    # Call the function
    display_chat_history()
    
    # Verify expected behaviors
    assert mock_streamlit["chat_message"].call_count == 6  # One for each message
    assert mock_streamlit["write"].call_count >= 2  # For text messages
    assert mock_streamlit["dataframe"].call_count >= 1  # For dataframe message
    assert mock_streamlit["pyplot"].call_count >= 1  # For figure message
    assert mock_streamlit["image"].call_count >= 1  # For image message


def test_display_chat_history_image_error(mock_streamlit, mock_session_state):
    """Test handling image errors when displaying chat history."""
    # Setup mock session state with image that will cause an error
    mock_session_state.chat_history = [
        {"role": "assistant", "type": "image", "content": "nonexistent.png"}
    ]
    
    # Setup image to raise an exception
    mock_streamlit["image"].side_effect = Exception("Image error")
    
    # Call the function
    display_chat_history()
    
    # Verify expected behaviors
    assert mock_streamlit["chat_message"].called
    assert mock_streamlit["error"].called  # Should display error


def test_handle_user_question_no_agent(mock_streamlit, mock_session_state):
    """Test handling a user question when no agent is available."""
    # Call the function
    handle_user_question("What is the average?")
    
    # Verify expected behaviors
    assert len(mock_session_state.chat_history) == 2  # User question + error message
    assert mock_session_state.chat_history[0]["role"] == "user"
    assert mock_session_state.chat_history[0]["content"] == "What is the average?"
    assert mock_session_state.chat_history[1]["role"] == "assistant"
    assert "Please upload a data file first" in mock_session_state.chat_history[1]["content"]
    assert mock_streamlit["error"].called


def test_handle_user_question_with_agent(mock_streamlit, mock_session_state, mock_agent):
    """Test handling a user question with a valid agent."""
    # Setup mock session state with agent
    mock_session_state.agent = mock_agent
    
    # Setup process_response to return text
    with patch('app.components.chat_components.process_response', return_value=("text", "The average is 42")):
        # Call the function
        handle_user_question("What is the average?")
        
        # Verify expected behaviors
        assert len(mock_session_state.chat_history) == 2  # User question + response
        assert mock_session_state.chat_history[0]["role"] == "user"
        assert mock_session_state.chat_history[0]["content"] == "What is the average?"
        assert mock_session_state.chat_history[1]["role"] == "assistant"
        assert mock_session_state.chat_history[1]["content"] == "The average is 42"
        mock_agent.chat.assert_called_once_with("What is the average?")


def test_handle_user_question_with_dataframe(mock_streamlit, mock_session_state, mock_agent, sample_dataframe):
    """Test handling a question that returns a dataframe."""
    # Setup mock session state with agent
    mock_session_state.agent = mock_agent
    
    # Setup process_response to return dataframe
    with patch('app.components.chat_components.process_response', return_value=("dataframe", sample_dataframe)):
        # Call the function
        handle_user_question("Show me the data")
        
        # Verify expected behaviors
        assert mock_streamlit["dataframe"].called
        assert mock_session_state.chat_history[1]["type"] == "dataframe"
        assert mock_session_state.chat_history[1]["content"] is sample_dataframe


def test_handle_user_question_with_figure(mock_streamlit, mock_session_state, mock_agent):
    """Test handling a question that returns a figure."""
    # Setup mock session state with agent
    mock_session_state.agent = mock_agent
    
    # Create a test figure
    fig, ax = plt.subplots()
    
    # Setup process_response to return figure
    with patch('app.components.chat_components.process_response', return_value=("figure", fig)), \
         patch('matplotlib.pyplot.close') as mock_close:
        # Call the function
        handle_user_question("Show me a chart")
        
        # Verify expected behaviors
        assert mock_streamlit["pyplot"].called
        assert mock_session_state.chat_history[1]["type"] == "figure"
        assert mock_session_state.chat_history[1]["content"] is fig
        mock_close.assert_called_once_with(fig)  # Should close the figure


def test_handle_user_question_with_image(mock_streamlit, mock_session_state, mock_agent):
    """Test handling a question that returns an image path."""
    # Setup mock session state with agent
    mock_session_state.agent = mock_agent
    
    # Test with existing image
    with patch('app.components.chat_components.process_response', return_value=("image", "test.png")), \
         patch('os.path.exists', return_value=True):
        # Call the function
        handle_user_question("Show me an image")
        
        # Verify expected behaviors
        assert mock_streamlit["image"].called
        assert mock_session_state.chat_history[1]["type"] == "image"
        assert mock_session_state.chat_history[1]["content"] == "test.png"
    
    # Test with non-existent image
    with patch('app.components.chat_components.process_response', return_value=("image", "nonexistent.png")), \
         patch('os.path.exists', return_value=False):
        # Call the function
        handle_user_question("Show me another image")
        
        # Verify expected behaviors
        assert mock_streamlit["error"].called
        assert mock_streamlit["write"].called  # Should write the raw response


def test_handle_user_question_with_exception(mock_streamlit, mock_session_state, mock_agent):
    """Test handling exceptions during question processing."""
    # Setup mock session state with agent
    mock_session_state.agent = mock_agent
    
    # Setup agent.chat to raise an exception
    mock_agent.chat.side_effect = Exception("Test exception")
    
    # Call the function
    handle_user_question("Cause an error")
    
    # Verify expected behaviors
    assert mock_streamlit["error"].called
    assert mock_session_state.chat_history[1]["type"] == "text"
    assert "Error generating response" in mock_session_state.chat_history[1]["content"]


def test_chat_interface(mock_streamlit, mock_session_state):
    """Test the chat interface."""
    # Setup mock session state
    mock_session_state.file_name = "test.csv"
    mock_session_state.chat_history = []
    
    # Test without user question
    mock_streamlit["chat_input"].return_value = None
    
    # Call the function
    chat_interface()
    
    # Verify expected behaviors
    assert mock_streamlit["subheader"].called
    assert "test.csv" in mock_streamlit["subheader"].call_args[0][0]
    
    # Test with user question
    mock_streamlit["chat_input"].return_value = "What is the average?"
    
    # Mock handle_user_question to prevent execution
    with patch('app.components.chat_components.handle_user_question') as mock_handle:
        # Call the function
        chat_interface()
        
        # Verify expected behaviors
        mock_handle.assert_called_once_with("What is the average?")


def test_reset_chat(mock_streamlit, mock_session_state):
    """Test resetting chat history."""
    # Setup mock session state
    mock_session_state.chat_history = ["message1", "message2"]
    
    # Test without clicking button
    mock_streamlit["button"].return_value = False
    
    # Call the function
    reset_chat()
    
    # Verify chat history is not cleared
    assert mock_session_state.chat_history == ["message1", "message2"]
    
    # Test with button click
    mock_streamlit["button"].return_value = True
    
    # Call the function
    with patch('streamlit.rerun') as mock_rerun:
        reset_chat()
        
        # Verify expected behaviors
        assert mock_session_state.chat_history == []
        assert mock_rerun.called 