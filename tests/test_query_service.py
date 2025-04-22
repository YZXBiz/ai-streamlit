"""Tests for the query service functionality."""
import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import streamlit as st

from app.services.query_service import QueryService
from app.models.agent import UserQuery, QueryResponse


class TestQueryService(unittest.TestCase):
    """Test cases for QueryService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a QueryService instance
        self.query_service = QueryService()
        
        # Create a mock DataFrame for testing
        self.test_data = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Apple', 'Banana', 'Orange', 'Grapes', 'Mango'],
            'category': ['Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit'],
            'price': [1.2, 0.5, 0.8, 2.5, 1.5],
            'quantity': [100, 150, 80, 60, 70]
        })
        
        # Mock the session state
        if not hasattr(st, 'session_state'):
            st.session_state = {}
        
        st.session_state.visualization_state = {}
        st.session_state.messages = [
            {"role": "user", "content": "Show me all products"},
            {"role": "assistant", "content": "Here are all the products."}
        ]

    @patch('app.services.query_service.data_chat_agent')
    @patch('app.services.query_service.DataService')
    def test_successful_query_execution(self, mock_data_service, mock_agent):
        """Test successful query processing and execution."""
        # Mock the agent response
        mock_response = QueryResponse(
            sql="SELECT * FROM products",
            explanation="Showing all products",
            visualization_type="table"
        )
        mock_agent.return_value = mock_response
        
        # Mock the data service
        mock_ds_instance = mock_data_service.return_value
        mock_ds_instance.get_schema_info.return_value = {"tables": ["products"]}
        mock_ds_instance.execute_query.return_value = (self.test_data, None)
        
        # Process a query
        result = self.query_service.process_query(
            "Show me all products", 
            self.test_data
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["sql"], "SELECT * FROM products")
        self.assertEqual(result["explanation"], "Showing all products")
        self.assertEqual(result["visualization_type"], "table")
        pd.testing.assert_frame_equal(result["result_data"], self.test_data)
        
        # Verify the agent was called with the correct parameters
        mock_agent.assert_called_once()
        called_query = mock_agent.call_args[0][0]
        self.assertIsInstance(called_query, UserQuery)
        self.assertEqual(called_query.query, "Show me all products")
        
        # Verify the data service was called
        mock_ds_instance.get_schema_info.assert_called_once()
        mock_ds_instance.execute_query.assert_called_once_with("SELECT * FROM products")

    @patch('app.services.query_service.data_chat_agent')
    @patch('app.services.query_service.DataService')
    def test_query_execution_error(self, mock_data_service, mock_agent):
        """Test handling of SQL execution errors."""
        # Mock the agent response
        mock_response = QueryResponse(
            sql="INVALID SQL QUERY",
            explanation="This will fail",
            visualization_type=None
        )
        mock_agent.return_value = mock_response
        
        # Mock the data service to return an error
        mock_ds_instance = mock_data_service.return_value
        mock_ds_instance.get_schema_info.return_value = {"tables": ["products"]}
        mock_ds_instance.execute_query.return_value = (None, "Syntax error in SQL")
        
        # Process a query
        result = self.query_service.process_query(
            "Run an invalid query", 
            self.test_data
        )
        
        # Verify the error handling
        self.assertFalse(result["success"])
        self.assertEqual(result["sql"], "INVALID SQL QUERY")
        self.assertIn("Error executing SQL", result["explanation"])
        
        # Verify the services were called
        mock_agent.assert_called_once()
        mock_ds_instance.execute_query.assert_called_once_with("INVALID SQL QUERY")

    @patch('app.services.query_service.data_chat_agent')
    @patch('app.services.query_service.DataService')
    def test_agent_no_sql_generated(self, mock_data_service, mock_agent):
        """Test handling when the agent cannot generate SQL."""
        # Mock the agent response with no SQL
        mock_response = QueryResponse(
            sql="",
            explanation="Could not understand the query",
            visualization_type=None
        )
        mock_agent.return_value = mock_response
        
        # Mock the data service
        mock_ds_instance = mock_data_service.return_value
        mock_ds_instance.get_schema_info.return_value = {"tables": ["products"]}
        
        # Process a query
        result = self.query_service.process_query(
            "A confusing query that can't be processed", 
            self.test_data
        )
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertIn("Could not generate SQL", result["explanation"])
        
        # Verify services called
        mock_agent.assert_called_once()
        # Execute query should not have been called
        mock_ds_instance.execute_query.assert_not_called()

    @patch('app.services.query_service.data_chat_agent')
    def test_exception_handling(self, mock_agent):
        """Test handling of exceptions during query processing."""
        # Mock the agent to raise an exception
        mock_agent.side_effect = Exception("Test exception")
        
        # Process a query
        result = self.query_service.process_query(
            "Query that causes an exception", 
            self.test_data
        )
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertIn("Error processing query", result["explanation"])
        self.assertIn("Test exception", result["explanation"])

    def test_visualization_suggestion(self):
        """Test the visualization suggestion logic."""
        # Test with a time series query
        result = self.query_service._suggest_visualization(
            "Show the trend of prices over time",
            pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'price': [10, 15, 12]
            })
        )
        self.assertEqual(result, "line")
        
        # Test with a comparison query
        result = self.query_service._suggest_visualization(
            "Compare the quantities of products",
            pd.DataFrame({
                'product': ['Apple', 'Banana', 'Orange'],
                'quantity': [100, 150, 80]
            })
        )
        self.assertEqual(result, "bar")
        
        # Test with a correlation query
        result = self.query_service._suggest_visualization(
            "Show the correlation between price and quantity",
            pd.DataFrame({
                'price': [1.2, 0.5, 0.8, 2.5, 1.5],
                'quantity': [100, 150, 80, 60, 70]
            })
        )
        self.assertEqual(result, "scatter")


if __name__ == '__main__':
    unittest.main() 