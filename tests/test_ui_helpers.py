"""
Tests for UI helper functions.

This module contains tests for the functions in ui/helpers.py.
"""

from unittest.mock import MagicMock

import pytest

from chatbot.ui.helpers import determine_complexity, enhance_query_with_context, table_exists


def test_determine_complexity_simple() -> None:
    """Test complexity determination with single table."""
    tables = ["users"]
    result = determine_complexity(tables)
    
    assert result == "simple"


def test_determine_complexity_advanced() -> None:
    """Test complexity determination with multiple tables."""
    # Exactly 2 tables
    tables = ["users", "orders"]
    result = determine_complexity(tables)
    assert result == "advanced"
    
    # More than 2 tables
    tables = ["users", "orders", "products"]
    result = determine_complexity(tables)
    assert result == "advanced"


def test_determine_complexity_empty() -> None:
    """Test complexity determination with empty list."""
    tables = []
    result = determine_complexity(tables)
    
    # Default to simple when no tables
    assert result == "simple"


def test_enhance_query_with_context_single_table() -> None:
    """Test query enhancement with single table."""
    query = "What is the average age?"
    tables = ["users"]
    
    enhanced = enhance_query_with_context(query, tables)
    
    # Verify enhanced query contains the table context
    assert "Available tables: users" in enhanced
    assert "Question: What is the average age?" in enhanced


def test_enhance_query_with_context_multiple_tables() -> None:
    """Test query enhancement with multiple tables."""
    query = "Find users in the Marketing department"
    tables = ["users", "departments", "projects"]
    
    enhanced = enhance_query_with_context(query, tables)
    
    # Verify enhanced query contains all tables
    assert "Available tables: users, departments, projects" in enhanced
    assert "Question: Find users in the Marketing department" in enhanced


def test_enhance_query_with_context_warning_message() -> None:
    """Test that enhancement includes warning about non-existent tables."""
    query = "Show my conversation history"
    tables = ["users"]
    
    enhanced = enhance_query_with_context(query, tables)
    
    # Verify warning about non-existent tables
    assert "do not reference tables that are not in the above list" in enhanced.lower()
    assert "conversation_history" in enhanced


def test_table_exists_success() -> None:
    """Test table_exists when table does exist."""
    mock_svc = MagicMock()
    # Mock successful query execution
    mock_svc.execute_query.return_value = True
    
    result = table_exists(mock_svc, "users")
    
    # Should return True
    assert result is True
    # Verify the correct query was executed
    mock_svc.execute_query.assert_called_once_with("SELECT 1 FROM users LIMIT 0")


def test_table_exists_failure() -> None:
    """Test table_exists when table doesn't exist."""
    mock_svc = MagicMock()
    # Mock query execution that raises an exception
    mock_svc.execute_query.side_effect = Exception("Table not found")
    
    result = table_exists(mock_svc, "nonexistent")
    
    # Should return False
    assert result is False
    # Verify the correct query was executed
    mock_svc.execute_query.assert_called_once_with("SELECT 1 FROM nonexistent LIMIT 0") 