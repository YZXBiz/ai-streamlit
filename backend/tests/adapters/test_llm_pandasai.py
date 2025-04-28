"""Tests for the PandasAI LLM adapter."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from app.adapters.llm_pandasai import PandasAILLMService


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "San Francisco", "Chicago"]
    })


class TestPandasAILLMService:
    """Test the PandasAILLMService adapter."""
    
    @pytest.mark.adapter
    def test_init(self):
        """Test initialization of PandasAILLMService."""
        llm_service = PandasAILLMService(api_key="test_key")
        assert llm_service.api_key == "test_key"
        assert llm_service.model == "gpt-3.5-turbo"  # Default model
        
        # Test with custom model
        llm_service = PandasAILLMService(api_key="test_key", model="gpt-4")
        assert llm_service.model == "gpt-4"
    
    @pytest.mark.adapter
    @patch("app.adapters.llm_pandasai.Agent")
    def test_get_text_response(self, mock_agent, sample_df):
        """Test getting a text response."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.chat.return_value = "The average age is 30"
        
        llm_service = PandasAILLMService(api_key="test_key")
        
        response = llm_service.get_text_response(
            prompt="What is the average age?",
            data=[sample_df]
        )
        
        # Verify the response
        assert response == "The average age is 30"
        
        # Verify Agent was initialized with the correct arguments
        mock_agent.assert_called_once()
        # Check that the LLM was created with the correct API key
        llm_config = mock_agent.call_args[1]["config"].llm
        assert llm_config.api_key == "test_key"
        assert llm_config.model == "gpt-3.5-turbo"
        
        # Verify the agent chat method was called with the correct prompt
        mock_agent_instance.chat.assert_called_once_with("What is the average age?")
    
    @pytest.mark.adapter
    @patch("app.adapters.llm_pandasai.Agent")
    def test_get_json_response(self, mock_agent, sample_df):
        """Test getting a JSON response."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.chat.return_value = '{"average_age": 30}'
        
        llm_service = PandasAILLMService(api_key="test_key")
        
        response = llm_service.get_json_response(
            prompt="What is the average age?",
            data=[sample_df]
        )
        
        # Verify the response is correctly parsed JSON
        assert isinstance(response, dict)
        assert response == {"average_age": 30}
        
        # Verify the agent chat method was called
        mock_agent_instance.chat.assert_called_once()
    
    @pytest.mark.adapter
    @patch("app.adapters.llm_pandasai.Agent")
    def test_get_chat_response(self, mock_agent, sample_df):
        """Test getting a chat response."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        # First message uses chat, follow-ups use follow_up
        mock_agent_instance.chat.return_value = "The average age is 30"
        mock_agent_instance.follow_up.return_value = "Alice is 25 years old"
        
        llm_service = PandasAILLMService(api_key="test_key")
        
        # Test initial message (should use chat)
        history = []
        response = llm_service.get_chat_response(
            prompt="What is the average age?",
            data=[sample_df],
            conversation_history=history
        )
        
        assert response == "The average age is 30"
        mock_agent_instance.chat.assert_called_once_with("What is the average age?")
        
        # Test follow-up message (should use follow_up)
        history = [
            {"role": "user", "content": "What is the average age?"},
            {"role": "assistant", "content": "The average age is 30"}
        ]
        
        response = llm_service.get_chat_response(
            prompt="How old is Alice?",
            data=[sample_df],
            conversation_history=history
        )
        
        assert response == "Alice is 25 years old"
        mock_agent_instance.follow_up.assert_called_once_with("How old is Alice?")
    
    @pytest.mark.adapter
    @patch("app.adapters.llm_pandasai.Agent")
    def test_get_document_response(self, mock_agent):
        """Test getting a document response."""
        # Setup mock
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.chat.return_value = "Document summary"
        
        llm_service = PandasAILLMService(api_key="test_key")
        
        response = llm_service.get_document_response(
            prompt="Summarize this document",
            document_text="This is a sample document with important information."
        )
        
        assert response == "Document summary"
        
        # Check that the agent was initialized with document content and not a DataFrame
        mock_agent.assert_called_once()
        # Verify prompt was passed to chat method
        mock_agent_instance.chat.assert_called_once_with("Summarize this document")         mock_agent_instance.chat.assert_called_once_with("Summarize this document") 