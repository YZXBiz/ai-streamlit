"""
Data service for handling data operations.

This module manages data loading, transformation, and storage.
"""
import os
import pandas as pd
import duckdb
import streamlit as st
from typing import Dict, Any, Optional, Tuple

# Import utilities for Snowflake connection
try:
    from utils.snowflake import SnowflakeConnector
except ImportError:
    # Define a mock class if snowflake utils are not available
    class SnowflakeConnector:
        def __init__(self, credentials: Dict[str, str]):
            self.credentials = credentials
            
        def execute_query(self, query: str) -> Optional[pd.DataFrame]:
            return None

class DataService:
    """Service for data operations including loading and transformations."""
    
    def __init__(self):
        """Initialize the data service."""
        # Initialize DuckDB connection
        self.db_conn = duckdb.connect(":memory:")
    
    def store_data(self, data: pd.DataFrame, source_type: str, source_name: str) -> bool:
        """
        Store data in session state and in DuckDB.
        
        Args:
            data: The DataFrame to store
            source_type: Type of source (file, snowflake)
            source_name: Name of the source
            
        Returns:
            Success flag
        """
        try:
            if data is None or data.empty:
                return False
            
            # Store in session state
            st.session_state.data = data
            
            # Get clean table name from source name
            table_name = self._clean_table_name(source_name)
            
            # Register with DuckDB
            self.db_conn.register(table_name, data)
            
            # Store metadata
            st.session_state.data_metadata = {
                "source_type": source_type,
                "source_name": source_name,
                "table_name": table_name,
                "row_count": len(data),
                "column_count": len(data.columns),
                "columns": list(data.columns)
            }
            
            return True
        except Exception as e:
            st.error(f"Error storing data: {e}")
            return False
    
    def query_snowflake(self, credentials: Dict[str, str], query: str) -> Optional[pd.DataFrame]:
        """
        Execute a query against Snowflake.
        
        Args:
            credentials: Snowflake connection credentials
            query: SQL query to execute
            
        Returns:
            DataFrame with query results or None if error
        """
        try:
            # Create Snowflake connector
            connector = SnowflakeConnector(credentials)
            
            # Execute query
            result_df = connector.execute_query(query)
            
            return result_df
        except Exception as e:
            st.error(f"Error querying Snowflake: {e}")
            return None
    
    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a SQL query against the DuckDB session.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (DataFrame with results, error message if any)
        """
        try:
            # Execute query against DuckDB
            result = self.db_conn.execute(query).fetchdf()
            return result, None
        except Exception as e:
            error_msg = str(e)
            return None, error_msg
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information for the loaded data.
        
        Returns:
            Dictionary with schema information
        """
        if 'data' not in st.session_state or st.session_state.data is None:
            return {}
        
        data = st.session_state.data
        
        # Build schema info dictionary
        schema_info = {
            'table_name': st.session_state.data_metadata.get('table_name', 'data'),
            'columns': {}
        }
        
        # Get column information
        for col in data.columns:
            dtype = str(data[col].dtype)
            
            # Determine the data type category
            if pd.api.types.is_numeric_dtype(data[col]):
                type_category = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                type_category = 'datetime'
            elif pd.api.types.is_bool_dtype(data[col]):
                type_category = 'boolean'
            else:
                type_category = 'string'
            
            # Get basic stats based on the data type
            stats = {}
            if type_category == 'numeric':
                stats = {
                    'min': data[col].min() if not data[col].empty else None,
                    'max': data[col].max() if not data[col].empty else None,
                    'mean': data[col].mean() if not data[col].empty else None,
                    'std': data[col].std() if not data[col].empty else None,
                }
            elif type_category == 'string':
                stats = {
                    'unique_count': data[col].nunique(),
                }
            
            # Add to schema info
            schema_info['columns'][col] = {
                'data_type': dtype,
                'type_category': type_category,
                'nullable': data[col].isna().any(),
                'unique_count': data[col].nunique(),
                'stats': stats
            }
        
        return schema_info
    
    def _clean_table_name(self, name: str) -> str:
        """
        Clean a string to be used as a table name.
        
        Args:
            name: The original name
            
        Returns:
            Cleaned name suitable for a table
        """
        # Remove file extension if present
        if '.' in name:
            name = name.split('.')[0]
        
        # Replace non-alphanumeric characters with underscore
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        
        # Ensure it starts with a letter
        if not clean_name[0].isalpha():
            clean_name = 'table_' + clean_name
        
        return clean_name.lower() 