"""Tests for the PandasAI adapter."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from backend.app.adapters.llm_pandasai import PandasAiAdapter
from backend.tests.utils.test_pandasai_utils import get_test_dataframe, MockAgent


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return get_test_dataframe()


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_dataframe.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


@patch("pandasai.Agent")
def test_init_adapter(mock_agent_class):
    """Test initialization of the PandasAI adapter."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Verify adapter has expected attributes
    assert hasattr(adapter, "get_agent")
    assert hasattr(adapter, "analyze")


@patch("pandasai.Agent")
def test_get_agent(mock_agent_class):
    """Test getting a PandasAI agent."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Get agent for a dataframe
    df = get_test_dataframe()
    agent = adapter.get_agent([df])
    
    # Assert agent is of expected type
    assert agent is not None
    assert agent is mock_agent
    
    # Agent should be configured with expected parameters
    mock_agent_class.assert_called_once()
    # Check if dfs are passed to agent
    args, kwargs = mock_agent_class.call_args
    assert "dfs" in kwargs
    assert len(kwargs["dfs"]) == 1


@patch("pandasai.Agent")
def test_analyze_with_file_path(mock_agent_class, temp_csv_file):
    """Test analyzing data with a file path."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock chat response
    mock_agent.chat.return_value = "Analysis result"
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Analyze with file path
    result = adapter.analyze(temp_csv_file, "What is the total sales?")
    
    # Assert expected behavior
    assert result == "Analysis result"
    mock_agent.chat.assert_called_once_with("What is the total sales?")


@patch("pandasai.Agent")
def test_analyze_with_dataframe(mock_agent_class):
    """Test analyzing data with a DataFrame."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock chat response
    mock_agent.chat.return_value = "Analysis result"
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Analyze with DataFrame
    df = get_test_dataframe()
    result = adapter.analyze(df, "What is the total sales?")
    
    # Assert expected behavior
    assert result == "Analysis result"
    mock_agent.chat.assert_called_once_with("What is the total sales?")


@patch("pandasai.Agent")
def test_follow_up_question(mock_agent_class):
    """Test asking follow-up questions."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock responses
    mock_agent.chat.return_value = "First answer"
    mock_agent.follow_up.return_value = "Follow-up answer"
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # First question
    df = get_test_dataframe()
    adapter.analyze(df, "What is the total sales?")
    
    # Follow-up question
    result = adapter.follow_up("How about by region?")
    
    # Assert expected behavior
    assert result == "Follow-up answer"
    mock_agent.follow_up.assert_called_once_with("How about by region?")


@patch("pandasai.Agent")
def test_cache_reuse(mock_agent_class):
    """Test that the adapter reuses cached agents."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Get agent twice with same DataFrame
    df = get_test_dataframe()
    agent1 = adapter.get_agent([df])
    
    # Reset mock to verify it's not called again
    mock_agent_class.reset_mock()
    
    agent2 = adapter.get_agent([df])
    
    # Assert agent is same instance and no new agent was created
    assert agent1 is agent2
    mock_agent_class.assert_not_called()


@patch("pandasai.Agent")
def test_error_handling(mock_agent_class):
    """Test error handling in the adapter."""
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock to raise an exception
    mock_agent.chat.side_effect = Exception("Test error")
    
    # Create adapter
    adapter = PandasAiAdapter()
    
    # Analyze with error
    df = get_test_dataframe()
    result = adapter.analyze(df, "This will cause an error")
    
    # Assert error message returned
    assert "error" in result.lower()
    assert "test error" in result.lower() 