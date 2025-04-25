"""
Tests for the main app module.

This module contains tests for the main app.py module.
"""

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from chatbot.app import main


@pytest.fixture
def mock_streamlit() -> None:
    """Mock all streamlit functions."""
    with patch("chatbot.app.st") as mock_st:
        # Set up basic streamlit functions
        mock_st.markdown = MagicMock()
        mock_st.set_page_config = MagicMock()
        mock_st.sidebar = MagicMock()
        mock_st.sidebar.markdown = MagicMock()
        mock_st.sidebar.expander = MagicMock()
        mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        mock_st.session_state = {}
        
        yield mock_st


@pytest.fixture
def setup_session_state() -> None:
    """Set up session state with a mock service."""
    mock_service = MagicMock()
    mock_service.tables = ["users", "products"]
    mock_service.initialize = MagicMock()
    mock_service.clear_data = MagicMock()
    
    if not hasattr(st, "session_state"):
        st.session_state = {}
    
    st.session_state.duckdb_service = mock_service
    st.session_state.last_result = None


@patch("chatbot.app.inject_styles")
@patch("chatbot.app.initialize_service")
@patch("chatbot.app.process_file_upload")
@patch("chatbot.app.display_chat_history")
@patch("chatbot.app.display_data_schema")
@patch("chatbot.app.display_results")
@patch("chatbot.app.determine_complexity")
@patch("chatbot.app.enhance_query_with_context")
def test_main_function_normal_flow(
    mock_enhance_query: MagicMock,
    mock_determine_complexity: MagicMock,
    mock_display_results: MagicMock,
    mock_display_data_schema: MagicMock,
    mock_display_chat_history: MagicMock,
    mock_process_file_upload: MagicMock,
    mock_initialize_service: MagicMock,
    mock_inject_styles: MagicMock,
    mock_streamlit: None,
    setup_session_state: None,
) -> None:
    """Test the main function normal execution flow."""
    # Set up file upload mock
    mock_process_file_upload.return_value = False  # No new files added
    
    # Set up complexity determination
    mock_determine_complexity.return_value = "advanced"
    
    # Call the main function
    main()
    
    # Verify styles were injected
    mock_inject_styles.assert_called_once()
    
    # Verify tab display functions were called
    mock_display_data_schema.assert_called_once()
    mock_display_chat_history.assert_called_once()


@patch("chatbot.app.inject_styles")
@patch("chatbot.app.initialize_service")
@patch("chatbot.app.st")
def test_main_service_initialization_success(
    mock_st: MagicMock,
    mock_initialize_service: MagicMock,
    mock_inject_styles: MagicMock,
) -> None:
    """Test service initialization success path."""
    # Set up mock service
    mock_service = MagicMock()
    mock_initialize_service.return_value = mock_service
    
    # Set up session state without service
    mock_st.session_state = {}
    
    # Call the main function
    main()
    
    # Verify service was initialized and stored in session state
    mock_initialize_service.assert_called_once()
    assert mock_st.session_state.duckdb_service == mock_service


@patch("chatbot.app.inject_styles")
@patch("chatbot.app.initialize_service")
@patch("chatbot.app.st")
def test_main_service_initialization_failure(
    mock_st: MagicMock,
    mock_initialize_service: MagicMock,
    mock_inject_styles: MagicMock,
) -> None:
    """Test service initialization failure path."""
    # Mock service initialization failure
    mock_initialize_service.return_value = None
    
    # Set up session state without service
    mock_st.session_state = {}
    
    # Call the main function
    main()
    
    # Verify error message and app stop
    mock_st.error.assert_called_with("Failed to initialize DuckDB service. Please check your OpenAI API key.")
    mock_st.stop.assert_called_once()


@patch("chatbot.app.inject_styles")
@patch("chatbot.app.st")
@patch("chatbot.app.process_file_upload")
def test_main_process_file_upload_rerun(
    mock_process_file_upload: MagicMock,
    mock_st: MagicMock,
    mock_inject_styles: MagicMock,
    setup_session_state: None,
) -> None:
    """Test app rerun after file upload."""
    # Mock a successful file upload
    mock_process_file_upload.return_value = True
    
    # Set up mock uploaded files
    mock_uploaded_files = [MagicMock()]
    mock_st.file_uploader.return_value = mock_uploaded_files
    
    # Call the main function
    main()
    
    # Verify file upload was processed
    mock_process_file_upload.assert_called_once_with(st.session_state.duckdb_service, mock_uploaded_files)
    
    # Verify app rerun
    mock_st.rerun.assert_called_once()


@patch("chatbot.app.inject_styles")
@patch("chatbot.app.st")
@patch("concurrent.futures.ThreadPoolExecutor")
@patch("chatbot.app.determine_complexity")
@patch("chatbot.app.enhance_query_with_context")
def test_main_query_processing_success(
    mock_enhance_query: MagicMock,
    mock_determine_complexity: MagicMock,
    mock_executor: MagicMock,
    mock_st: MagicMock,
    mock_inject_styles: MagicMock,
    setup_session_state: None,
) -> None:
    """Test successful query processing."""
    # Mock form submission
    form_submit_button = MagicMock()
    form_submit_button.return_value = True
    mock_st.form_submit_button = form_submit_button
    
    # Mock text input
    mock_st.text_input.return_value = "What is the average age of users?"
    
    # Mock complexity determination
    mock_determine_complexity.return_value = "simple"
    
    # Mock query enhancement
    mock_enhance_query.return_value = "Enhanced: What is the average age of users?"
    
    # Mock thread executor
    mock_future = MagicMock()
    mock_future.result.return_value = {"success": True, "data": "35 years"}
    mock_executor_instance = MagicMock()
    mock_executor_instance.submit.return_value = mock_future
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    
    # Call the main function
    main()
    
    # Verify query processing
    service = st.session_state.duckdb_service
    mock_executor_instance.submit.assert_called_once_with(
        service.process_query, "Enhanced: What is the average age of users?", "natural_language", "simple"
    )
    mock_future.result.assert_called_once_with(timeout=20)
    

@patch("chatbot.app.inject_styles")
@patch("chatbot.app.st")
@patch("concurrent.futures.ThreadPoolExecutor")
@patch("chatbot.app.determine_complexity")
@patch("chatbot.app.enhance_query_with_context")
def test_main_query_processing_timeout(
    mock_enhance_query: MagicMock,
    mock_determine_complexity: MagicMock, 
    mock_executor: MagicMock,
    mock_st: MagicMock,
    mock_inject_styles: MagicMock,
    setup_session_state: None,
) -> None:
    """Test query processing timeout."""
    # Mock form submission
    form_submit_button = MagicMock()
    form_submit_button.return_value = True
    mock_st.form_submit_button = form_submit_button
    
    # Mock text input
    mock_st.text_input.return_value = "Complex query that times out"
    
    # Mock complexity determination
    mock_determine_complexity.return_value = "advanced"
    
    # Mock query enhancement
    mock_enhance_query.return_value = "Enhanced: Complex query that times out"
    
    # Mock thread executor with timeout
    mock_future = MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError()
    mock_executor_instance = MagicMock()
    mock_executor_instance.submit.return_value = mock_future
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    
    # Call the main function
    main()
    
    # Verify timeout error
    mock_st.error.assert_called_with(
        "Query processing timed out after 20 seconds. Please try a simpler query or check your data."
    ) 