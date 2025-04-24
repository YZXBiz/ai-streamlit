"""
Base DuckDB service module.

This module provides core DuckDB functionality for loading data and executing SQL queries.
"""

from typing import Any, Dict, List

import duckdb
import pandas as pd

from flat_chatbot.logger import get_logger

logger = get_logger(__name__)


class DuckDBService:
    """Base DuckDB service for data storage and querying."""

    def __init__(self, db_path: str | None = None):
        """Initialize the DuckDB service.

        Args:
            db_path: Optional path to DuckDB database file. If None, an in-memory
                    database will be used.
        """
        self.conn = duckdb.connect(database=db_path if db_path else ":memory:")
        self.tables: List[str] = []

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with results
        """
        result = self.conn.execute(query).fetchdf()
        return result

    def load_dataframe(
        self, df: pd.DataFrame | List[pd.DataFrame], table_name: str | List[str]
    ) -> bool:
        """Load a DataFrame or list of DataFrames into DuckDB with transaction support.

        Args:
            df: DataFrame or list of DataFrames to load
            table_name: Name of the table to create or list of table names

        Returns:
            bool: True if successful
        """
        # Handle single DataFrame case
        if isinstance(df, pd.DataFrame) and isinstance(table_name, str):
            return self._load_single_dataframe(df, table_name)

        # Handle list of DataFrames case
        elif isinstance(df, list) and isinstance(table_name, list):
            if len(df) != len(table_name):
                logger.error("Number of DataFrames and table names must match")
                return False

            try:
                self.conn.execute("BEGIN TRANSACTION")
                for single_df, single_table in zip(df, table_name, strict=True):
                    if not isinstance(single_df, pd.DataFrame):
                        raise TypeError(f"Expected DataFrame, got {type(single_df)}")
                    self.conn.register(single_table, single_df)
                    self.tables.append(single_table)
                self.conn.execute("COMMIT")
                return True
            except Exception as e:
                self.conn.execute("ROLLBACK")
                logger.error("Failed to load DataFrames: %s", e)
                return False
        else:
            logger.error(
                "Invalid parameter types. Both df and table_name must be either single values or lists."
            )
            return False

    def _load_single_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Load a single DataFrame into DuckDB.

        Args:
            df: DataFrame to load
            table_name: Name of the table to create

        Returns:
            bool: True if successful
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.register(table_name, df)
            self.tables.append(table_name)
            self.conn.execute("COMMIT")
            return True
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error("Failed to load DataFrame: %s", e)
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information for all tables.

        Returns:
            Dict with schema information
        """
        schema_info = {
            "tables": self.tables,
            "columns": {},
        }

        for table in self.tables:
            # Get column information
            columns = self.execute_query(f"DESCRIBE {table}")
            schema_info["columns"][table] = columns["column_name"].tolist()

        return schema_info

    def clear_data(self) -> bool:
        """Clear all data from the service.

        Returns:
            bool: True if successful
        """
        try:
            # Drop all tables and views
            for table in self.tables:
                try:
                    # Try to drop as view
                    self.conn.execute(f"DROP VIEW IF EXISTS {table}")
                except Exception as e:
                    # Ignore catalog/type mismatch errors
                    if "is of type Table, trying to drop type View" not in str(e):
                        logger.warning(f"Error dropping view {table}: {e}")

                try:
                    # Try to drop as table
                    self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                except Exception as e:
                    # Ignore catalog/type mismatch errors
                    if "is of type View, trying to drop type Table" not in str(e):
                        logger.warning(f"Error dropping table {table}: {e}")

            self.tables = []
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False

    def __del__(self) -> None:
        """Clean up database connection."""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception:
            pass

    def load_file_directly(self, file_path: str, table_name: str) -> bool:
        """Load a file directly into DuckDB using native loaders.

        Bypasses pandas DataFrame conversion to avoid Arrow compatibility issues.

        Args:
            file_path: Path to the file to load
            table_name: Name of the table to create

        Returns:
            bool: True if successful
        """
        try:
            file_extension = file_path.split(".")[-1].lower()
            self.conn.execute("BEGIN TRANSACTION")

            if file_extension == "csv":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')"
                )
            elif file_extension == "parquet":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
                )
            elif file_extension in ["xls", "xlsx"]:
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_excel('{file_path}')"
                )
            elif file_extension == "json":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_json_auto('{file_path}')"
                )
            else:
                logger.error("Unsupported file extension: %s", file_extension)
                self.conn.execute("ROLLBACK")
                return False

            # Add to tables list if successful
            self.tables.append(table_name)
            self.conn.execute("COMMIT")
            logger.info("Successfully loaded %s into table %s", file_path, table_name)
            return True

        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error("Failed to load file directly: %s", e)
            return False
