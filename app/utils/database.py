"""
Database utility functions.

This module provides utilities for DuckDB operations.
"""
import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

class DuckDBManager:
    """Manager for DuckDB connection and operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB connection.
        
        Args:
            db_path: Path to DuckDB database file or ":memory:" for in-memory database
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError(
                "DuckDB dependencies not available. "
                "Please install with: uv add duckdb"
            )
        
        # Use in-memory database by default
        if db_path is None:
            db_path = os.environ.get("DUCKDB_PATH", ":memory:")
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.tables = {}
    
    def register_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """
        Register a DataFrame as a table in DuckDB.
        
        Args:
            table_name: Name for the table
            df: DataFrame to register
            
        Returns:
            Success flag
        """
        try:
            self.conn.register(table_name, df)
            self.tables[table_name] = {
                "columns": list(df.columns),
                "rows": len(df)
            }
            return True
        except Exception as e:
            print(f"Error registering table: {e}")
            return False
    
    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (DataFrame with results, error message if any)
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return result, None
        except Exception as e:
            error_msg = str(e)
            return None, error_msg
    
    def get_table_names(self) -> List[str]:
        """
        Get list of registered table names.
        
        Returns:
            List of table names
        """
        return list(self.tables.keys())
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with schema information or None if not found
        """
        if table_name not in self.tables:
            return None
        
        try:
            # Query DuckDB for schema information
            query = f"DESCRIBE {table_name}"
            schema_df, error = self.execute_query(query)
            
            if error or schema_df is None:
                return None
            
            # Build schema dictionary
            schema_info = {
                "table_name": table_name,
                "columns": {}
            }
            
            for _, row in schema_df.iterrows():
                col_name = row['column_name']
                col_type = row['column_type']
                
                # Determine column type category
                type_category = "string"
                if any(numeric_type in col_type.lower() for numeric_type in 
                       ['int', 'float', 'double', 'decimal', 'numeric']):
                    type_category = "numeric"
                elif 'date' in col_type.lower() or 'time' in col_type.lower():
                    type_category = "datetime"
                elif 'bool' in col_type.lower():
                    type_category = "boolean"
                
                schema_info["columns"][col_name] = {
                    "data_type": col_type,
                    "type_category": type_category,
                    "nullable": True  # DuckDB DESCRIBE doesn't provide nullability info
                }
            
            return schema_info
        except Exception as e:
            print(f"Error getting schema for table {table_name}: {e}")
            return None
    
    def close(self):
        """Close the DuckDB connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None 