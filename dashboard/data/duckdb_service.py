"""DuckDB service for in-memory data processing and SQL queries.

This module provides functionality to store and query data using DuckDB,
an in-memory SQL database that's optimized for analytics.
"""

import pandas as pd
import duckdb
import streamlit as st
from typing import Optional

from dashboard.settings import get_settings


class DuckDBService:
    """Service for in-memory data storage and querying using DuckDB.
    
    This class provides methods to load data, execute SQL queries, and
    retrieve schema information for use in the chatbot assistant.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the DuckDB service.
        
        Args:
            db_path: Optional path to the DuckDB database file or ":memory:" for in-memory database.
                    If None, the path will be loaded from settings.
        """
        settings = get_settings()
        self.db_path = db_path if db_path is not None else settings.duckdb.db_path
        self.max_preview_rows = settings.duckdb.max_rows_preview
        
        self.conn = duckdb.connect(self.db_path)
        self.tables = {}  # Track registered tables
        
    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Load a pandas DataFrame into DuckDB.
        
        Args:
            df: The pandas DataFrame to load
            table_name: Name of the table to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Register the dataframe as a table
            self.conn.register(table_name, df)
            
            # Keep track of the table
            self.tables[table_name] = {
                "columns": list(df.columns),
                "row_count": len(df),
                "col_count": len(df.columns),
                "column_types": {col: str(df[col].dtype) for col in df.columns}
            }
            
            return True
        except Exception as e:
            st.error(f"Error loading data into DuckDB: {str(e)}")
            return False
            
    def execute_query(self, query: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a SQL query against the in-memory database.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (result DataFrame, error message)
              - If successful, returns (DataFrame, None)
              - If error occurs, returns (None, error_message)
        """
        try:
            result = self.conn.execute(query).fetch_df()
            return result, None
        except Exception as e:
            error_msg = str(e)
            return None, error_msg
    
    def get_schema_info(self) -> dict:
        """Get information about the database schema.
        
        Returns:
            Dictionary containing tables and their column information
        """
        return {
            "tables": list(self.tables.keys()),
            "schema": self.tables
        }
        
    def get_table_preview(self, table_name: str, limit: int = None) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get a preview of data in a table.
        
        Args:
            table_name: Name of the table to preview
            limit: Maximum number of rows to return. If None, uses the value from settings.
            
        Returns:
            Tuple of (preview DataFrame, error message)
        """
        if limit is None:
            limit = self.max_preview_rows
            
        if table_name not in self.tables:
            return None, f"Table '{table_name}' not found"
            
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
    
    def clear_data(self):
        """Clear all data from the in-memory database."""
        # Close the existing connection and create a new one
        self.conn.close()
        self.conn = duckdb.connect(self.db_path)
        self.tables = {} 