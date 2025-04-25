"""
Tests for file upload functionality.

This module contains tests for the process_file_upload function.
"""

import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

from chatbot.ui.file_upload import process_file_upload
from chatbot.ui.helpers import table_exists


@pytest.fixture
def mock_uploaded_file() -> MagicMock:
    """Return a mock uploaded file."""
    mock_file = MagicMock()
    mock_file.name = "test_data.csv"
    mock_file.size = 1024
    mock_file.getbuffer.return_value = b"id,name\n1,test"
    return mock_file


class TestFileUpload:
    """Test file upload functionality."""

    @pytest.fixture
    def mock_session_state_dict(self) -> dict:
        """Return a mock session state dict."""
        return {"processed_files": set()}

    @pytest.fixture
    def mock_spinner(self) -> MagicMock:
        """Return a mock spinner context manager."""
        mock = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=mock)
        mock.return_value.__exit__ = MagicMock(return_value=None)
        return mock

    def test_no_files_provided(self, mock_streamlit: MagicMock) -> None:
        """Test behavior when no files are provided."""
        # Set up mocks
        mock_service = MagicMock()

        # Call the function
        result = process_file_upload(mock_service, None)

        # Verify expected behavior
        assert result is False
        mock_service.load_file_directly.assert_not_called()
        mock_service.initialize.assert_not_called()

    def test_skip_already_processed_file(
        self, mock_uploaded_file: MagicMock, mock_streamlit: MagicMock
    ) -> None:
        """Test skipping files that were already processed."""
        # Set up mocks
        mock_service = MagicMock()

        # Add file ID to processed list
        file_id = f"{mock_uploaded_file.name}_{mock_uploaded_file.size}"
        mock_streamlit.session_state = {"processed_files": {file_id}}

        # Call the function
        result = process_file_upload(mock_service, [mock_uploaded_file])

        # Verify expected behavior
        assert result is False
        mock_service.load_file_directly.assert_not_called()
        mock_service.initialize.assert_not_called()

    @patch("chatbot.ui.helpers.table_exists")
    def test_skip_existing_table(
        self, mock_table_exists: MagicMock, mock_uploaded_file: MagicMock, mock_streamlit: MagicMock
    ) -> None:
        """Test skipping files with tables that already exist."""
        # Set up mocks
        mock_service = MagicMock()
        mock_table_exists.return_value = True
        mock_streamlit.session_state = {"processed_files": set()}

        # Call the function
        result = process_file_upload(mock_service, [mock_uploaded_file])

        # Verify expected behavior
        assert result is False
        mock_table_exists.assert_called_once_with(mock_service, "test_data")
        mock_service.load_file_directly.assert_not_called()
        mock_service.initialize.assert_not_called()

        # Verify the file was marked as processed
        file_id = f"{mock_uploaded_file.name}_{mock_uploaded_file.size}"
        assert file_id in mock_streamlit.session_state["processed_files"]

    @patch("chatbot.ui.helpers.table_exists")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_successful_file_processing(
        self,
        mock_unlink: MagicMock,
        mock_tempfile: MagicMock,
        mock_table_exists: MagicMock,
        mock_uploaded_file: MagicMock,
        mock_streamlit: MagicMock,
    ) -> None:
        """Test successful file processing."""
        # Set up mocks
        mock_service = MagicMock()
        mock_service.load_file_directly.return_value = True
        mock_table_exists.return_value = False

        # Set up temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_file.csv"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Initialize processed files
        mock_streamlit.session_state = {"processed_files": set()}

        # Call the function
        result = process_file_upload(mock_service, [mock_uploaded_file])

        # Verify expected behavior
        assert result is True
        mock_service.load_file_directly.assert_called_once_with(mock_temp_file.name, "test_data")
        mock_unlink.assert_called_once_with(mock_temp_file.name)
        mock_service.initialize.assert_called_once()

        # Verify the file was marked as processed
        file_id = f"{mock_uploaded_file.name}_{mock_uploaded_file.size}"
        assert file_id in mock_streamlit.session_state["processed_files"]

    @patch("chatbot.ui.helpers.table_exists")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_failed_file_loading(
        self,
        mock_unlink: MagicMock,
        mock_tempfile: MagicMock,
        mock_table_exists: MagicMock,
        mock_uploaded_file: MagicMock,
        mock_streamlit: MagicMock,
    ) -> None:
        """Test handling of file load failures."""
        # Set up mocks
        mock_service = MagicMock()
        mock_service.load_file_directly.return_value = False  # Simulate failure
        mock_table_exists.return_value = False

        # Set up temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_file.csv"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Initialize processed files
        mock_streamlit.session_state = {"processed_files": set()}

        # Call the function
        result = process_file_upload(mock_service, [mock_uploaded_file])

        # Verify expected behavior
        assert result is False
        mock_service.load_file_directly.assert_called_once_with(mock_temp_file.name, "test_data")
        mock_unlink.assert_called_once_with(mock_temp_file.name)
        mock_service.initialize.assert_not_called()

        # Verify the file was NOT marked as processed
        file_id = f"{mock_uploaded_file.name}_{mock_uploaded_file.size}"
        assert file_id not in mock_streamlit.session_state["processed_files"]

    @patch("chatbot.ui.helpers.table_exists")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_multiple_files(
        self,
        mock_unlink: MagicMock,
        mock_tempfile: MagicMock,
        mock_table_exists: MagicMock,
        mock_uploaded_file: MagicMock,
        mock_streamlit: MagicMock,
    ) -> None:
        """Test processing multiple files."""
        # Set up mocks
        mock_service = MagicMock()
        mock_service.load_file_directly.return_value = True
        mock_table_exists.return_value = False

        # Create a second mock file
        mock_file2 = MagicMock()
        mock_file2.name = "other_data.csv"
        mock_file2.size = 2048
        mock_file2.getbuffer.return_value = b"id,value\n1,x\n2,y"

        # Set up temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_file.csv"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Initialize processed files
        mock_streamlit.session_state = {"processed_files": set()}

        # Call the function
        result = process_file_upload(mock_service, [mock_uploaded_file, mock_file2])

        # Verify expected behavior
        assert result is True
        assert mock_service.load_file_directly.call_count == 2
        mock_service.initialize.assert_called_once()

        # Verify both files were marked as processed
        file_id1 = f"{mock_uploaded_file.name}_{mock_uploaded_file.size}"
        file_id2 = f"{mock_file2.name}_{mock_file2.size}"
        assert file_id1 in mock_streamlit.session_state["processed_files"]
        assert file_id2 in mock_streamlit.session_state["processed_files"]
