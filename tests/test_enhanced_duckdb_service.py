"""
Tests for the EnhancedDuckDBService class.

This module contains tests for the EnhancedDuckDBService class.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas import DataFrame

from chatbot.services import EnhancedDuckDBService


def test_init_basic_properties(enhanced_duckdb_service: EnhancedDuckDBService) -> None:
    """Test basic properties after initialization."""
    assert enhanced_duckdb_service.conn is not None
    assert enhanced_duckdb_service.tables == []
    assert enhanced_duckdb_service.table_schemas == {}
    assert enhanced_duckdb_service.simple_query_engine is None
    assert enhanced_duckdb_service.advanced_query_engine is None
    assert enhanced_duckdb_service.table_index is None
    assert hasattr(enhanced_duckdb_service, "memory")


def test_init_chat_memory_simple(mock_embed_model: Any) -> None:
    """Test initialization with simple chat memory."""
    service = EnhancedDuckDBService(
        embed_model=mock_embed_model,
        memory_type="simple",
        token_limit=2000,
    )
    
    assert hasattr(service, "memory")
    assert service.memory.__class__.__name__ == "ChatMemoryBuffer"


def test_init_chat_memory_summary(mock_embed_model: Any) -> None:
    """Test initialization with summary chat memory."""
    service = EnhancedDuckDBService(
        embed_model=mock_embed_model,
        memory_type="summary",
        token_limit=2000,
    )
    
    assert hasattr(service, "memory")
    assert service.memory.__class__.__name__ == "ChatSummaryMemoryBuffer"


def test_get_table_schemas(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test _get_table_schemas method."""
    # Reset table_schemas to test the method
    loaded_enhanced_service.table_schemas = {}
    
    # Call the method
    loaded_enhanced_service._get_table_schemas()
    
    # Verify table schemas were retrieved
    assert "users" in loaded_enhanced_service.table_schemas
    assert "departments" in loaded_enhanced_service.table_schemas
    
    # Verify schema structure
    users_schema = loaded_enhanced_service.table_schemas["users"]
    assert users_schema["name"] == "users"
    assert "columns" in users_schema
    assert "types" in users_schema
    assert len(users_schema["columns"]) == len(users_schema["types"])
    assert "name" in users_schema["columns"]
    assert "id" in users_schema["columns"]


def test_init_query_engines(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test initialization of query engines."""
    # Reset query engines to test the method
    loaded_enhanced_service.simple_query_engine = None
    loaded_enhanced_service.advanced_query_engine = None
    loaded_enhanced_service.table_index = None
    
    # Call initialize to set up query engines
    loaded_enhanced_service.initialize()
    
    # Verify query engines were initialized
    assert loaded_enhanced_service.simple_query_engine is not None
    assert loaded_enhanced_service.advanced_query_engine is not None
    assert loaded_enhanced_service.table_index is not None


def test_initialize_no_tables(enhanced_duckdb_service: EnhancedDuckDBService) -> None:
    """Test initialize with no tables."""
    # Call initialize with no tables
    enhanced_duckdb_service.initialize()
    
    # Verify query engines were not initialized
    assert enhanced_duckdb_service.simple_query_engine is None
    assert enhanced_duckdb_service.advanced_query_engine is None
    assert enhanced_duckdb_service.table_index is None


@patch("time.time", return_value=12345)
def test_execute_duckdb_query(
    mock_time: MagicMock, enhanced_duckdb_service: EnhancedDuckDBService, sample_df: DataFrame
) -> None:
    """Test _execute_duckdb_query method."""
    # Load sample data
    enhanced_duckdb_service.load_dataframe(sample_df, "users")
    
    # Execute query
    result = enhanced_duckdb_service._execute_duckdb_query("SELECT * FROM users")
    
    # Verify result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_df)
    assert list(result.columns) == list(sample_df.columns)


def test_process_sql_query(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test process_query with SQL query type."""
    # Process SQL query
    query = "SELECT * FROM users WHERE age > 30"
    result = loaded_enhanced_service.process_query(query, "sql")
    
    # Verify result structure
    assert result["success"] is True
    assert "data" in result
    assert "sql_query" in result
    assert result["sql_query"] == query
    assert "raw_data" in result
    
    # Verify data
    assert isinstance(result["raw_data"], list)
    assert len(result["raw_data"]) == 3  # Users with age > 30
    assert result["raw_data"][0]["name"] in ["Charlie", "David", "Eve"]


def test_process_nl_query_simple(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test process_query with natural language query (simple mode)."""
    # Mock the simple query engine to return a fixed result
    mock_result = MagicMock()
    mock_result.response = "Average age is 35"
    mock_result.metadata = {"sql_query": "SELECT AVG(age) FROM users"}
    loaded_enhanced_service.simple_query_engine = MagicMock()
    loaded_enhanced_service.simple_query_engine.query.return_value = mock_result
    
    # Process natural language query
    query = "What is the average age of users?"
    result = loaded_enhanced_service.process_query(query, "natural_language", "simple")
    
    # Verify result structure
    assert result["success"] is True
    assert result["data"] == "Average age is 35"
    assert result["sql_query"] == "SELECT AVG(age) FROM users"
    
    # Verify memory was updated
    assert "What is the average age of users?" in loaded_enhanced_service.get_chat_history()
    assert "Average age is 35" in loaded_enhanced_service.get_chat_history()


def test_process_nl_query_advanced(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test process_query with natural language query (advanced mode)."""
    # Mock the advanced query engine to return a fixed result
    mock_result = MagicMock()
    mock_result.response = "Users in Marketing: Charlie"
    mock_result.metadata = {"sql_query": "SELECT u.name FROM users u JOIN departments d ON u.id = d.id WHERE d.department = 'Marketing'"}
    loaded_enhanced_service.advanced_query_engine = MagicMock()
    loaded_enhanced_service.advanced_query_engine.query.return_value = mock_result
    
    # Process natural language query
    query = "Which users are in the Marketing department?"
    result = loaded_enhanced_service.process_query(query, "natural_language", "advanced")
    
    # Verify result structure
    assert result["success"] is True
    assert result["data"] == "Users in Marketing: Charlie"
    assert "sql_query" in result
    assert "Marketing" in result["sql_query"]
    
    # Verify memory was updated
    assert "Which users are in the Marketing department?" in loaded_enhanced_service.get_chat_history()
    assert "Users in Marketing: Charlie" in loaded_enhanced_service.get_chat_history()


def test_process_query_error(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test process_query with error handling."""
    # Process invalid query
    query = "SELECT * FROM nonexistent_table"
    result = loaded_enhanced_service.process_query(query, "sql")
    
    # Verify error handling
    assert result["success"] is False
    assert "error" in result
    assert "nonexistent_table" in result["error"]
    
    # Error should still be in memory
    history = loaded_enhanced_service.get_chat_history()
    assert "Error" in history
    assert "nonexistent_table" in history


def test_get_chat_history(enhanced_duckdb_service: EnhancedDuckDBService) -> None:
    """Test get_chat_history method."""
    # Initial history should be empty
    initial_history = enhanced_duckdb_service.get_chat_history()
    assert initial_history == "No chat history available."
    
    # Add some messages to memory
    enhanced_duckdb_service.memory.put("user", "What is the average age?")
    enhanced_duckdb_service.memory.put("assistant", "The average age is 35.")
    
    # Get updated history
    updated_history = enhanced_duckdb_service.get_chat_history()
    assert "What is the average age?" in updated_history
    assert "The average age is 35." in updated_history


def test_clear_chat_history(enhanced_duckdb_service: EnhancedDuckDBService) -> None:
    """Test clear_chat_history method."""
    # Add some messages to memory
    enhanced_duckdb_service.memory.put("user", "Test message")
    enhanced_duckdb_service.memory.put("assistant", "Test response")
    
    # Verify messages are in history
    assert "Test message" in enhanced_duckdb_service.get_chat_history()
    
    # Clear history
    enhanced_duckdb_service.clear_chat_history()
    
    # Verify history is cleared
    assert enhanced_duckdb_service.get_chat_history() == "No chat history available."


def test_change_chat_session(enhanced_duckdb_service: EnhancedDuckDBService) -> None:
    """Test change_chat_session method."""
    # Add message to current session
    enhanced_duckdb_service.memory.put("user", "Session 1 message")
    
    # Change to new session
    enhanced_duckdb_service.change_chat_session("new_session")
    
    # Verify new session is empty
    assert enhanced_duckdb_service.get_chat_history() == "No chat history available."
    
    # Add message to new session
    enhanced_duckdb_service.memory.put("user", "Session 2 message")
    
    # Change back to original session
    enhanced_duckdb_service.change_chat_session("streamlit_user")
    
    # Verify original session has original message
    assert "Session 1 message" in enhanced_duckdb_service.get_chat_history()
    assert "Session 2 message" not in enhanced_duckdb_service.get_chat_history()


def test_clear_data_override(loaded_enhanced_service: EnhancedDuckDBService) -> None:
    """Test clear_data override method."""
    # Verify tables exist before clearing
    assert loaded_enhanced_service.tables
    assert loaded_enhanced_service.table_schemas
    
    # Call clear_data
    result = loaded_enhanced_service.clear_data()
    
    # Verify data is cleared
    assert result is True
    assert loaded_enhanced_service.tables == []
    assert loaded_enhanced_service.table_schemas == {}
    assert loaded_enhanced_service.simple_query_engine is None
    assert loaded_enhanced_service.advanced_query_engine is None 