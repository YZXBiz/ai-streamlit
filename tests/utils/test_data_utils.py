"""Tests for the data utilities."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.utils.data_utils import (
    display_data_info,
    initialize_agent,
    load_dataframe,
    process_response,
)


def test_initialize_agent_with_api_key(sample_dataframe):
    """Test initializing AI agent with a provided API key."""
    with (
        patch("app.utils.data_utils.OpenAI") as mock_openai,
        patch("app.utils.data_utils.DataFrame") as mock_dataframe,
        patch("app.utils.data_utils.Agent") as mock_agent,
    ):
        # Setup mocks
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        mock_pai_df = MagicMock()
        mock_dataframe.return_value = mock_pai_df
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Call the function
        agent, error = initialize_agent(sample_dataframe, "test_api_key")

        # Verify expected behaviors
        assert error is None
        assert agent == mock_agent_instance
        mock_openai.assert_called_once_with(api_token="test_api_key")
        mock_dataframe.assert_called_once_with(sample_dataframe)
        mock_agent.assert_called_once()


def test_initialize_agent_without_api_key(sample_dataframe):
    """Test initializing AI agent without providing an API key."""
    # Test when no API key is provided or in environment
    with patch.dict(os.environ, {}, clear=True):
        agent, error = initialize_agent(sample_dataframe)
        assert agent is None
        assert error == "Missing OpenAI API Key"

    # Test when API key is in environment
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "env_api_key"}),
        patch("app.utils.data_utils.OpenAI") as mock_openai,
        patch("app.utils.data_utils.DataFrame") as mock_dataframe,
        patch("app.utils.data_utils.Agent") as mock_agent,
    ):
        # Setup mocks
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Call the function
        agent, error = initialize_agent(sample_dataframe)

        # Verify expected behaviors
        assert error is None
        assert agent == mock_agent_instance
        mock_openai.assert_called_once_with(api_token="env_api_key")


def test_initialize_agent_exception(sample_dataframe):
    """Test handling exceptions during agent initialization."""
    with patch("app.utils.data_utils.OpenAI") as mock_openai:
        # Setup mock to raise exception
        mock_openai.side_effect = Exception("Test exception")

        # Call the function
        agent, error = initialize_agent(sample_dataframe, "test_api_key")

        # Verify expected behaviors
        assert agent is None
        assert error == "Test exception"


def test_load_dataframe_csv(mock_uploaded_file):
    """Test loading a CSV file into a dataframe."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        # Setup mock temp file
        mock_temp = MagicMock()
        mock_temp.name = "test.csv"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # Setup mock read_csv
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            mock_read_csv.return_value = mock_df

            # Mock os.unlink to prevent actual file deletion
            with patch("os.unlink"):
                # Set file name to have .csv extension
                mock_uploaded_file.name = "test.csv"

                # Call the function
                df, error = load_dataframe(mock_uploaded_file)

                # Verify expected behaviors
                assert error is None
                assert df is mock_df
                mock_temp.write.assert_called_once_with(mock_uploaded_file.getvalue())
                mock_read_csv.assert_called_once_with(mock_temp.name)


def test_load_dataframe_excel(mock_uploaded_file):
    """Test loading an Excel file into a dataframe."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        # Setup mock temp file
        mock_temp = MagicMock()
        mock_temp.name = "test.xlsx"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # Setup mock read_excel
        with patch("pandas.read_excel") as mock_read_excel:
            mock_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            mock_read_excel.return_value = mock_df

            # Mock os.unlink to prevent actual file deletion
            with patch("os.unlink"):
                # Set file name to have .xlsx extension
                mock_uploaded_file.name = "test.xlsx"

                # Call the function
                df, error = load_dataframe(mock_uploaded_file)

                # Verify expected behaviors
                assert error is None
                assert df is mock_df
                mock_temp.write.assert_called_once_with(mock_uploaded_file.getvalue())
                mock_read_excel.assert_called_once_with(mock_temp.name)


def test_load_dataframe_no_file():
    """Test handling when no file is provided."""
    df, error = load_dataframe(None)
    assert df is None
    assert error == "No file uploaded"


def test_load_dataframe_unsupported_format(mock_uploaded_file):
    """Test handling unsupported file formats."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        # Setup mock temp file
        mock_temp = MagicMock()
        mock_temp.name = "test.txt"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # Mock os.unlink to prevent actual file deletion
        with patch("os.unlink"):
            # Set file name to have .txt extension
            mock_uploaded_file.name = "test.txt"

            # Call the function
            df, error = load_dataframe(mock_uploaded_file)

            # Verify expected behaviors
            assert df is None
            assert error == "Unsupported file format"


def test_load_dataframe_exception(mock_uploaded_file):
    """Test handling exceptions during file loading."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        # Setup mock to raise exception
        mock_temp_file.side_effect = Exception("Test exception")

        # Call the function
        df, error = load_dataframe(mock_uploaded_file)

        # Verify expected behaviors
        assert df is None
        assert error == "Test exception"


def test_process_response_dataframe(sample_dataframe):
    """Test processing a dataframe response."""
    # Call the function with a dataframe
    response_type, content = process_response(sample_dataframe)

    # Verify expected behaviors
    assert response_type == "dataframe"
    assert content is sample_dataframe


def test_process_response_figure():
    """Test processing a matplotlib figure response."""
    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    # Call the function with a figure
    response_type, content = process_response(fig)

    # Verify expected behaviors
    assert response_type == "figure"
    assert content is fig

    # Clean up
    plt.close(fig)


def test_process_response_image_exists():
    """Test processing an image path response when file exists."""
    with (
        patch("os.path.exists", return_value=True),
        patch("PIL.Image.open") as mock_image_open,
        patch("matplotlib.pyplot.subplots") as mock_subplots,
    ):
        # Setup mocks
        mock_img = MagicMock()
        mock_image_open.return_value = mock_img
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Call the function with an image path
        response_type, content = process_response("data/charts/test.png")

        # Verify expected behaviors
        assert response_type == "figure"
        assert content is mock_fig
        mock_image_open.assert_called_once_with("data/charts/test.png")
        mock_ax.imshow.assert_called_once_with(mock_img)
        mock_ax.axis.assert_called_once_with("off")


def test_process_response_image_not_exists():
    """Test processing an image path response when file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        # Call the function with a non-existent image path
        response_type, content = process_response("data/charts/nonexistent.png")

        # Verify expected behaviors
        assert response_type == "text"
        assert content == "Chart file not found: data/charts/nonexistent.png"


def test_process_response_text():
    """Test processing a text response."""
    # Call the function with a text string
    response_type, content = process_response("This is a text response")

    # Verify expected behaviors
    assert response_type == "text"
    assert content == "This is a text response"


def test_process_response_image_exception():
    """Test processing an image path when exception occurs during image loading."""
    with patch("os.path.exists", return_value=True), patch("PIL.Image.open") as mock_image_open:
        # Setup mock to raise exception
        mock_image_open.side_effect = Exception("Test exception")

        # Call the function with an image path
        response_type, content = process_response("data/charts/test.png")

        # Verify expected behaviors
        assert response_type == "image"
        assert content == "data/charts/test.png"


def test_display_data_info(mock_streamlit, sample_dataframe):
    """Test displaying dataframe information."""
    # Call the function
    display_data_info(sample_dataframe)

    # Verify expected behaviors
    assert mock_streamlit["subheader"].call_count >= 3  # Should call subheader at least 3 times
    assert mock_streamlit["dataframe"].call_count >= 3  # Should display at least 3 dataframes
    assert mock_streamlit["write"].called  # Should write something
