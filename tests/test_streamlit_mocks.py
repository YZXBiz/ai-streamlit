"""
Test streamlit mocks to ensure they work correctly.

These tests confirm that our streamlit mocks behave as expected
in our testing environment.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the tests directory to the path to enable relative imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from conftest import MockSt


def test_mock_streamlit_basic_methods() -> None:
    """Test basic streamlit mock methods."""
    mock_st = MockSt()

    # Test basic methods
    mock_st.markdown("test")
    mock_st.write("test")
    mock_st.info("test")
    mock_st.error("test")
    mock_st.success("test")
    mock_st.warning("test")

    # Verify calls
    mock_st.markdown.assert_called_with("test")
    mock_st.write.assert_called_with("test")
    mock_st.info.assert_called_with("test")
    mock_st.error.assert_called_with("test")
    mock_st.success.assert_called_with("test")
    mock_st.warning.assert_called_with("test")


def test_mock_streamlit_container_context_managers() -> None:
    """Test streamlit mock context managers."""
    mock_st = MockSt()

    # Test container context manager
    with mock_st.container():
        mock_st.write("inside container")

    # Test expander context manager
    with mock_st.expander("Expand me"):
        mock_st.write("inside expander")

    # Test spinner context manager
    with mock_st.spinner("Loading..."):
        mock_st.write("inside spinner")

    # Verify enter/exit were called
    mock_st.container.assert_called_once()
    mock_st.container.return_value.__enter__.assert_called_once()
    mock_st.container.return_value.__exit__.assert_called_once()

    mock_st.expander.assert_called_once_with("Expand me")
    mock_st.expander.return_value.__enter__.assert_called_once()
    mock_st.expander.return_value.__exit__.assert_called_once()

    mock_st.spinner.assert_called_once_with("Loading...")
    mock_st.spinner.return_value.__enter__.assert_called_once()
    mock_st.spinner.return_value.__exit__.assert_called_once()


def test_mock_streamlit_form() -> None:
    """Test streamlit mock form functionality."""
    mock_st = MockSt()

    # Test form context manager with submit button
    with mock_st.form("my-form"):
        mock_st.text_input("Name")
        submitted = mock_st.form_submit_button("Submit")

    # Verify form methods were called
    mock_st.form.assert_called_once_with("my-form")
    mock_st.text_input.assert_called_once_with("Name")
    mock_st.form_submit_button.assert_called_once_with("Submit")
    assert submitted is False  # Default return value


def test_mock_streamlit_session_state() -> None:
    """Test streamlit mock session state."""
    mock_st = MockSt()

    # Test session state
    mock_st.session_state["key1"] = "value1"
    mock_st.session_state["key2"] = 42

    # Verify session state behaves as expected
    assert mock_st.session_state["key1"] == "value1"
    assert mock_st.session_state["key2"] == 42
    assert "key3" not in mock_st.session_state


@patch("streamlit.markdown")
def test_mock_streamlit_patching(mock_markdown: MagicMock) -> None:
    """Test patching of streamlit module."""
    # Import after patching to use the mock
    import streamlit as st

    # Use the module
    st.markdown("test patched")

    # Verify the mock was used
    mock_markdown.assert_called_with("test patched")


def test_mock_streamlit_layout() -> None:
    """Test streamlit mock layout methods."""
    mock_st = MockSt()

    # Test columns
    col1, col2, col3 = mock_st.columns(3)
    col1.write("Column 1")
    col2.write("Column 2")
    col3.write("Column 3")

    # Test tabs - get all tabs in a list
    tabs = mock_st.tabs(["Tab 1", "Tab 2"])
    for i, tab in enumerate(tabs):
        tab.write(f"Tab {i + 1} content")

    # Verify columns and tabs methods were called
    mock_st.columns.assert_called_once_with(3)
    mock_st.tabs.assert_called_once_with(["Tab 1", "Tab 2"])
    assert len(mock_st.columns.return_value) == 3
