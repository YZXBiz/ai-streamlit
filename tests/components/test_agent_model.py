import pytest
import pandas as pd
from unittest import mock

from app.models.agent_model import AgentModel
from pandasai import Agent, DataFrame


def test_create_agent_single_dataframe():
    """Test creating an agent with a single dataframe."""
    # Create test data
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    
    # Mock the OpenAI LLM
    with mock.patch("app.models.agent_model.OpenAI") as mock_openai:
        mock_llm = mock.MagicMock()
        mock_openai.return_value = mock_llm
        
        # Mock the Agent creation
        with mock.patch("app.models.agent_model.Agent") as mock_agent_class:
            mock_agent = mock.MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Create the agent
            agent_model = AgentModel()
            agent, error = agent_model.create_agent(df, "fake-api-key")
            
            # Verify the agent was created with a single dataframe
            assert error is None
            assert agent is mock_agent
            
            # Check the dataframe was passed correctly
            mock_agent_class.assert_called_once()
            _, kwargs = mock_agent_class.call_args
            
            # Should be a single DataFrame
            assert "dfs" in kwargs
            assert isinstance(kwargs["dfs"], DataFrame)


def test_create_agent_multiple_dataframes():
    """Test creating an agent with multiple dataframes."""
    # Create test data
    customers_df = pd.DataFrame({
        "customer_id": [1, 2], 
        "name": ["John", "Jane"]
    })
    
    orders_df = pd.DataFrame({
        "order_id": [101, 102], 
        "customer_id": [1, 2], 
        "amount": [100, 200]
    })
    
    # Create dictionary of dataframes
    dfs = {
        "customers": customers_df,
        "orders": orders_df
    }
    
    # Mock the OpenAI LLM
    with mock.patch("app.models.agent_model.OpenAI") as mock_openai:
        mock_llm = mock.MagicMock()
        mock_openai.return_value = mock_llm
        
        # Mock the Agent creation
        with mock.patch("app.models.agent_model.Agent") as mock_agent_class:
            mock_agent = mock.MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Create the agent
            agent_model = AgentModel()
            agent, error = agent_model.create_agent(dfs, "fake-api-key")
            
            # Verify the agent was created with multiple dataframes
            assert error is None
            assert agent is mock_agent
            
            # Check the dataframes were passed correctly
            mock_agent_class.assert_called_once()
            _, kwargs = mock_agent_class.call_args
            
            # Should be a dictionary of DataFrames
            assert "dfs" in kwargs
            assert isinstance(kwargs["dfs"], dict)
            assert len(kwargs["dfs"]) == 2
            assert "customers" in kwargs["dfs"]
            assert "orders" in kwargs["dfs"]
            assert isinstance(kwargs["dfs"]["customers"], DataFrame)
            assert isinstance(kwargs["dfs"]["orders"], DataFrame)


def test_create_agent_missing_api_key():
    """Test agent creation fails with missing API key."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    
    agent_model = AgentModel()
    agent, error = agent_model.create_agent(df, "")
    
    assert agent is None
    assert error == "Missing OpenAI API Key" 