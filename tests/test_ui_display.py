"""
Tests for UI display functions.

This module contains tests for the display functions in ui/display.py.
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

from chatbot.ui.display import display_chat_history, display_data_schema, display_results


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a mock database service for testing."""
    mock_svc = MagicMock()

    # Mock table schemas data
    table_schemas = {
        "users": ["id INTEGER", "name TEXT", "age INTEGER", "city TEXT"],
        "orders": ["id INTEGER", "user_id INTEGER", "product TEXT", "amount DECIMAL"],
    }

    # Sample table data for previews
    table_data = {
        "users": pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "city": ["New York", "Chicago", "Los Angeles"],
            }
        ),
        "orders": pd.DataFrame(
            {
                "id": [101, 102, 103],
                "user_id": [1, 1, 2],
                "product": ["Laptop", "Mouse", "Monitor"],
                "amount": [1200.00, 25.50, 350.00],
            }
        ),
    }

    # Set up mock methods
    mock_svc.get_table_schemas.return_value = table_schemas
    mock_svc.tables = list(table_schemas.keys())

    def mock_execute_query(query):
        """Mock query execution."""
        if "users" in query.lower():
            return table_data["users"]
        elif "orders" in query.lower():
            return table_data["orders"]
        return pd.DataFrame()

    mock_svc.execute_duckdb_query = mock_execute_query
    mock_svc.get_chat_history.return_value = "User: How many users?\nAssistant: There are 3 users."

    return mock_svc


@patch("streamlit.markdown")
@patch("streamlit.write")
@patch("streamlit.dataframe")
@patch("streamlit.download_button")
def test_display_results_success(
    mock_download_button: MagicMock,
    mock_dataframe: MagicMock,
    mock_write: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Test display_results with successful query."""
    # Create a successful result
    result = {
        "success": True,
        "data": "Average age is 35",
        "sql_query": "SELECT AVG(age) FROM users",
        "raw_data": [{"avg_age": 35}],
    }

    # Display results
    display_results(result)

    # Verify data display
    mock_write.assert_called_with("Average age is 35")

    # Verify SQL query display in expander
    mock_markdown.assert_any_call(
        "<div class='section-header'>Response</div>", unsafe_allow_html=True
    )

    # Verify that raw data was displayed as DataFrame
    mock_dataframe.assert_called()

    # Verify download buttons were created
    assert mock_download_button.call_count >= 1


@patch("streamlit.markdown")
@patch("streamlit.error")
def test_display_results_error(
    mock_error: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Test display_results with error result."""
    # Create an error result
    result = {"success": False, "error": "Table not found"}

    # Display results
    display_results(result)

    # Verify error display
    mock_markdown.assert_called_with(
        """<div class="error-box">
                <strong>Error:</strong> Table not found
                </div>""",
        unsafe_allow_html=True,
    )


@patch("streamlit.session_state")
@patch("streamlit.markdown")
@patch("streamlit.write")
@patch("streamlit.info")
def test_display_results_empty_data(
    mock_info: MagicMock,
    mock_write: MagicMock,
    mock_markdown: MagicMock,
    mock_session_state: MagicMock,
) -> None:
    """Test display_results with empty data."""
    # Create a result with empty raw_data
    result = {
        "success": True,
        "data": "No data found",
        "sql_query": "SELECT * FROM users WHERE age > 100",
        "raw_data": [],
    }

    # Display results
    display_results(result)

    # Verify warning for empty data was shown
    mock_info.assert_called_with("No data to display")


@patch("streamlit.markdown")
@patch("streamlit.info")
@patch("streamlit.expander")
@patch("streamlit.dataframe")
def test_display_data_schema(
    mock_dataframe: MagicMock,
    mock_expander: MagicMock,
    mock_info: MagicMock,
    mock_markdown: MagicMock,
    mock_service: MagicMock,
) -> None:
    """Test display_data_schema function."""
    # Mock the return value of get_table_schemas
    mock_service.get_table_schemas.return_value = {
        "users": ["id INTEGER", "name TEXT", "age INTEGER"]
    }

    mock_service.execute_duckdb_query.return_value = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]}
    )

    # Display schema
    display_data_schema(mock_service)

    # Verify section header
    mock_markdown.assert_called_with(
        "<div class='section-header'>Database Schema</div>", unsafe_allow_html=True
    )

    # Verify expanders were created for each table
    assert mock_expander.call_count == 2  # One for each table

    # Verify data preview was displayed for each table
    assert mock_dataframe.call_count >= 2  # At least once for each table


@patch("streamlit.markdown")
@patch("streamlit.info")
def test_display_data_schema_no_tables(
    mock_info: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Test display_data_schema with no tables."""
    # Create mock service with no tables
    mock_svc = MagicMock()
    mock_svc.tables = []

    # Display schema
    display_data_schema(mock_svc)

    # Verify info message
    mock_info.assert_called_with(
        "No tables loaded. Please upload data files to view schema information."
    )

    # Verify no section header
    mock_markdown.assert_not_called()


@patch("streamlit.markdown")
@patch("streamlit.button")
@patch("streamlit.success")
@patch("streamlit.rerun")
@patch("streamlit.download_button")
@patch("streamlit.text_area")
def test_display_chat_history(
    mock_text_area: MagicMock,
    mock_download_button: MagicMock,
    mock_rerun: MagicMock,
    mock_success: MagicMock,
    mock_button: MagicMock,
    mock_markdown: MagicMock,
    mock_service: MagicMock,
) -> None:
    """Test display_chat_history function."""
    # Set up button to simulate clicking "Clear Chat History"
    mock_button.return_value = True

    # Display chat history
    display_chat_history(mock_service)

    # Verify section header
    mock_markdown.assert_called_with(
        "<div class='section-header'>Conversation History</div>", unsafe_allow_html=True
    )

    # Verify clear history button was clicked and history was cleared
    mock_service.clear_chat_history.assert_called_once()
    mock_success.assert_called_with("Chat history cleared")
    mock_rerun.assert_called_once()

    # Text area and download button won't be called because of the mock rerun
    mock_text_area.assert_not_called()
    mock_download_button.assert_not_called()


@patch("streamlit.markdown")
@patch("streamlit.button")
@patch("streamlit.info")
def test_display_chat_history_empty(
    mock_info: MagicMock,
    mock_button: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    """Test display_chat_history with empty history."""
    # Create mock service with empty history
    mock_svc = MagicMock()
    mock_svc.get_chat_history.return_value = ""

    # Set up button to simulate not clicking
    mock_button.return_value = False

    # Display chat history
    display_chat_history(mock_svc)

    # Verify section header
    mock_markdown.assert_called_with(
        "<div class='section-header'>Conversation History</div>", unsafe_allow_html=True
    )

    # Verify info message for empty history
    mock_info.assert_called_with("No conversation history yet. Start asking questions!")
