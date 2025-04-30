import pytest
import pandas as pd
from unittest import mock

from app.models.agent_model import AgentModel
from pandasai import Agent


@pytest.fixture
def sample_dataframes():
    """Create sample dataframes for testing."""
    # Customer data
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "country": ["USA", "Canada", "UK"]
    })
    
    # Order data with foreign key to customers
    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 2, 1, 3],
        "product": ["Widget", "Gadget", "Widget", "Tool"],
        "amount": [100.50, 200.75, 50.25, 300.00],
        "date": ["2023-01-15", "2023-01-20", "2023-02-10", "2023-02-25"]
    })
    
    return {
        "customers": customers,
        "orders": orders
    }


@mock.patch("pandasai.Agent.chat")
@mock.patch("pandasai_openai.OpenAI")
def test_cross_table_query(mock_openai, mock_chat, sample_dataframes):
    """Test that the agent can answer questions spanning multiple tables."""
    # Setup the agent with multiple dataframes
    agent_model = AgentModel()
    mock_llm = mock.MagicMock()
    mock_openai.return_value = mock_llm
    
    # Mock the response 
    mock_response = mock.MagicMock()
    mock_response.type = "dataframe"
    
    # Create a result DataFrame that shows joined data (as if PandasAI executed a join)
    result_df = pd.DataFrame({
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "country": ["USA", "Canada", "UK"],
        "total_spent": [150.75, 200.75, 300.00]
    })
    mock_response.value = result_df
    mock_chat.return_value = mock_response
    
    # Create the agent
    agent, error = agent_model.create_agent(sample_dataframes, "fake-api-key")
    
    # Test with a question that requires joining the tables
    question = "What is the total amount spent by each customer? Include their name and country."
    response = agent_model.process_question(agent, question, True)
    
    # Verify chat was called with the cross-table question
    mock_chat.assert_called_once_with(question)
    
    # Verify the response has the expected shape (combined data)
    assert response.type == "dataframe"
    assert response.value is result_df
    
    # The result should have customer names and order amounts (joined data)
    assert "name" in response.value.columns
    assert "total_spent" in response.value.columns


@mock.patch("pandasai.Agent.chat")
@mock.patch("pandasai_openai.OpenAI")
def test_table_reference_in_query(mock_openai, mock_chat, sample_dataframes):
    """Test queries that explicitly reference table names."""
    # Setup the agent with multiple dataframes
    agent_model = AgentModel()
    mock_llm = mock.MagicMock()
    mock_openai.return_value = mock_llm
    
    # Mock the response
    mock_response = mock.MagicMock()
    mock_response.type = "string"
    mock_response.value = "The analysis shows that customer John Doe from USA has placed 2 orders."
    mock_chat.return_value = mock_response
    
    # Create the agent
    agent, error = agent_model.create_agent(sample_dataframes, "fake-api-key")
    
    # Test with a question that explicitly references tables
    question = "Join the customers and orders tables and tell me how many orders John Doe has placed."
    response = agent_model.process_question(agent, question, True)
    
    # Verify chat was called with the explicit table reference question
    mock_chat.assert_called_once_with(question)
    
    # Verify we got a string response
    assert response.type == "string"
    assert "John Doe" in response.value
    assert "2 orders" in response.value 