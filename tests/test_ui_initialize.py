"""
Tests for UI initialize module.

This module contains tests for the initialize_service function in ui/initialize.py.
"""

from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding

from chatbot.ui.initialize import initialize_service


@pytest.fixture
def mock_config() -> None:
    """Set up mocked config for tests."""
    with patch("chatbot.ui.initialize.config") as mock_config:
        # Set up a valid API key
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "valid-api-key"
        mock_config.OPENAI_API_KEY = mock_secret

        # Set up other config values
        mock_config.DB_PATH = ":memory:"
        mock_config.MEMORY_TYPE = "simple"
        mock_config.MEMORY_TOKEN_LIMIT = 3000

        yield mock_config


def test_initialize_service_success(mock_streamlit: MagicMock) -> None:
    """Test successful service initialization."""
    with patch("chatbot.ui.initialize.load_dotenv") as mock_load_dotenv:
        with patch("chatbot.ui.initialize.OpenAIEmbedding") as mock_embedding_class:
            with patch("chatbot.ui.initialize.EnhancedDuckDBService") as mock_service_class:
                with patch("chatbot.ui.initialize.config") as mock_config:
                    # Set up mock embedding instance
                    mock_embedding = MagicMock()
                    mock_embedding_class.return_value = mock_embedding

                    # Set up mock service instance
                    mock_service = MagicMock()
                    mock_service_class.return_value = mock_service

                    # Set up config mock
                    mock_secret = MagicMock()
                    mock_secret.get_secret_value.return_value = "valid-api-key"
                    mock_config.OPENAI_API_KEY = mock_secret
                    mock_config.DB_PATH = ":memory:"
                    mock_config.MEMORY_TYPE = "simple"
                    mock_config.MEMORY_TOKEN_LIMIT = 3000

                    # Call the function
                    result = initialize_service()

                    # Verify result is the mocked service
                    assert result == mock_service

                    # Verify dotenv was loaded
                    assert mock_load_dotenv.called

                    # Verify embedding was initialized
                    assert mock_embedding_class.called

                    # Verify service was initialized
                    assert mock_service_class.called


def test_initialize_service_missing_api_key(mock_streamlit: MagicMock) -> None:
    """Test initialization with missing API key."""
    with patch("chatbot.ui.initialize.config") as mock_config:
        # Set up a missing API key
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = ""
        mock_config.OPENAI_API_KEY = mock_secret

        # Call the function
        result = initialize_service()

        # Verify result is None
        assert result is None

        # Verify error was displayed
        assert mock_streamlit.error.called
        found_error = False
        for call in mock_streamlit.error.call_args_list:
            args, kwargs = call
            if args and "OPENAI_API_KEY not found" in args[0]:
                found_error = True
                break
        assert found_error, "API key error message not displayed"


def test_initialize_service_invalid_api_key(mock_streamlit: MagicMock) -> None:
    """Test initialization with invalid API key (raises ValidationError)."""
    with patch("chatbot.ui.initialize.config") as mock_config:
        # Set up config to raise ValueError
        mock_config.OPENAI_API_KEY = MagicMock()
        mock_config.OPENAI_API_KEY.get_secret_value.side_effect = ValueError("Invalid API key")

        # Call the function
        result = initialize_service()

        # Verify result is None
        assert result is None

        # Verify error was displayed
        assert mock_streamlit.error.called
        found_error = False
        for call in mock_streamlit.error.call_args_list:
            args, kwargs = call
            if args and "invalid" in args[0].lower():
                found_error = True
                break
        assert found_error, "Invalid API key error message not displayed"
