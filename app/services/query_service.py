"""
Query service for handling natural language queries.

This module manages query generation, execution, and explanation.
"""
import os
from typing import Dict, Any, Optional, List, Tuple, Union

import pandas as pd
import streamlit as st

from models.agent import data_chat_agent, UserQuery, QueryResponse
from services.data_service import DataService

class QueryService:
    """
    Service for processing natural language queries.
    
    This class handles the conversion of natural language queries into SQL,
    executes the generated SQL, and provides explanations of the results.
    """
    
    def __init__(self) -> None:
        """
        Initialize the query service.
        
        Creates a new DataService instance for database operations.
        """
        self.data_service = DataService()
    
    def process_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a natural language query on the provided data.
        
        Parameters
        ----------
        query : str
            The natural language query to process
        data : pd.DataFrame
            The DataFrame to query against
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - success (bool): Whether the query was processed successfully
            - sql (str, optional): The generated SQL query
            - explanation (str): Human-readable explanation of the results or error
            - result_data (pd.DataFrame, optional): DataFrame with query results
            - visualization_type (str, optional): Suggested visualization for the results
        """
        try:
            # Build context from schema info
            schema_info = self.data_service.get_schema_info()
            
            # Create context for the agent
            context = {
                "schema": schema_info,
                "data_sample": data.head(5).to_dict(orient="records"),
                "visualization_state": st.session_state.visualization_state,
                "conversation_history": self._get_conversation_history()
            }
            
            # Create user query object
            user_query = UserQuery(
                query=query,
                context=context
            )
            
            # Process with the agent
            response = data_chat_agent(user_query)
            
            # Execute the generated SQL
            if hasattr(response, 'sql') and response.sql:
                result_data, error = self.data_service.execute_query(response.sql)
                
                if error:
                    return {
                        "success": False,
                        "explanation": f"Error executing SQL: {error}",
                        "sql": response.sql
                    }
                
                # Build result dictionary
                result = {
                    "success": True,
                    "sql": response.sql,
                    "explanation": response.explanation,
                    "result_data": result_data
                }
                
                # Add visualization type if available
                if hasattr(response, 'visualization_type') and response.visualization_type:
                    result["visualization_type"] = response.visualization_type
                
                return result
            else:
                return {
                    "success": False,
                    "explanation": "Could not generate SQL for your query. Please try rephrasing."
                }
                
        except Exception as e:
            return {
                "success": False,
                "explanation": f"Error processing query: {str(e)}"
            }
    
    def _get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history from session state.
        
        Returns
        -------
        List[Dict[str, str]]
            List of conversation messages, each containing:
            - role (str): Either 'user' or 'assistant'
            - content (str): The message content
        """
        if "messages" not in st.session_state:
            return []
        
        # Convert messages to format needed by the agent
        history = []
        for msg in st.session_state.messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return history
    
    def _suggest_visualization(self, query: str, result: pd.DataFrame) -> Optional[str]:
        """
        Suggest an appropriate visualization for the query result.
        
        Parameters
        ----------
        query : str
            The original natural language query
        result : pd.DataFrame
            The query result DataFrame
            
        Returns
        -------
        Optional[str]
            Suggested visualization type or None if no visualization is appropriate.
            Possible values include: 'bar', 'line', 'scatter', 'pie', 'histogram', 'table'
        
        Examples
        --------
        >>> df = pd.DataFrame({'date': ['2023-01-01', '2023-01-02'], 'value': [10, 20]})
        >>> service = QueryService()
        >>> service._suggest_visualization("Show trend over time", df)
        'line'
        """
        # Check if the query result is suitable for visualization
        if result is None or result.empty or len(result.columns) < 1:
            return None
        
        # Check for aggregation or time series queries
        query_lower = query.lower()
        has_numeric_columns = any(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)
        
        if ("trend" in query_lower or "over time" in query_lower) and has_numeric_columns:
            return "line"
        
        if ("compare" in query_lower or "comparison" in query_lower) and has_numeric_columns:
            return "bar"
        
        if "distribution" in query_lower and has_numeric_columns:
            return "histogram"
        
        if "correlation" in query_lower and has_numeric_columns and len(result.columns) >= 2:
            return "scatter"
        
        # Default suggestions based on result structure
        if len(result.columns) == 1:
            if pd.api.types.is_numeric_dtype(result[result.columns[0]]):
                return "bar"
            else:
                return "pie"
        
        if len(result.columns) == 2:
            if all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns):
                return "scatter"
            elif any(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns):
                return "bar"
        
        # For larger result sets with numeric columns
        if has_numeric_columns and len(result.columns) > 2:
            return "table"
        
        return None 