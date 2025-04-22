"""
DuckDB service for storing and querying data.

This module provides a service for interacting with DuckDB, an in-process
SQL OLAP database management system.
"""

import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


class DuckDBService:
    """Service for interacting with DuckDB database."""

    def __init__(self, db_path: str | None = None):
        """Initialize the DuckDB service.

        Args:
            db_path: Optional path to DuckDB database file. If None, the path from
                     settings will be used. If the path is ":memory:", an in-memory
                     database will be used.
        """
        if db_path:
            # Use persistent storage
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(db_file))
            logger.info(f"Connected to DuckDB database at {db_path}")
        else:
            # Use in-memory database
            self.conn = duckdb.connect(":memory:")
            logger.info("Connected to in-memory DuckDB database")

        # Track tables loaded into the database
        self.tables: dict[str, dict[str, Any]] = {}

    def connect(self) -> None:
        """Connect to the DuckDB database."""
        try:
            # Create directory for database if it doesn't exist and not using in-memory
            if self.conn.db_path != ":memory:":
                os.makedirs(os.path.dirname(self.conn.db_path), exist_ok=True)

            # Connect to database
            self.conn = duckdb.connect(self.conn.db_path)
            logger.info(f"Connected to DuckDB at {self.conn.db_path}")
        except Exception:
            logger.error("Error connecting to DuckDB", exc_info=True)
            # Fall back to in-memory database
            self.conn.db_path = ":memory:"
            self.conn = duckdb.connect(self.conn.db_path)
            logger.info("Falling back to in-memory DuckDB database")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Closed DuckDB connection")

    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Load a pandas DataFrame into DuckDB.

        Args:
            df: The DataFrame to load
            table_name: The name of the table to create

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Drop table if it already exists
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Register DataFrame as a table
            self.conn.register(table_name, df)

            # Create a persistent table from the registered view
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}")

            # Store metadata about the table
            self.tables[table_name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            }

            logger.info(f"Loaded DataFrame with {len(df)} rows into table '{table_name}'")
            return True
        except Exception:
            logger.error(f"Error loading DataFrame into table '{table_name}'", exc_info=True)
            return False

    def execute_query(self, query: str) -> pd.DataFrame | None:
        """Execute a SQL query and return the results as a DataFrame.

        Args:
            query: The SQL query to execute

        Returns:
            Optional[pd.DataFrame]: DataFrame with query results, or None if an error occurred
        """
        try:
            logger.info(f"Executing query: {query}")
            return self.conn.execute(query).fetchdf()
        except Exception:
            logger.error("Error executing query", exc_info=True)
            return None

    def get_schema_info(self) -> dict[str, Any]:
        """Get information about the database schema.

        Returns:
            Dict with tables and their column information
        """
        schema_info: dict[str, Any] = {
            "tables": list(self.tables.keys()),
            "columns": {},
        }

        # Get column information for each table
        for table_name, table_info in self.tables.items():
            schema_info["columns"][table_name] = table_info["columns"]

        return schema_info

    def clear_data(self) -> bool:
        """Clear all data from the database.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Close existing connection
            self.close()

            # If using a file-based database, delete the file
            if self.conn.db_path != ":memory:" and os.path.exists(self.conn.db_path):
                os.remove(self.conn.db_path)
                logger.info(f"Deleted DuckDB file at {self.conn.db_path}")

            # Reconnect to create a fresh database
            self.connect()

            # Drop all tables
            for table_name in self.tables.keys():
                try:
                    self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                except Exception:
                    logger.error(f"Error dropping table {table_name}", exc_info=True)

            # Reset tables dictionary
            self.tables = {}
            logger.info("All data cleared from database")

            return True
        except Exception:
            logger.error("Error clearing DuckDB data", exc_info=True)
            return False

    def get_data_statistics(self, table_name: str) -> dict[str, Any]:
        """Get statistical information about a table.

        Args:
            table_name: The name of the table to get statistics for

        Returns:
            Dict containing statistical information about the table
        """
        if not table_name or table_name not in self.tables:
            return {"error": f"Table '{table_name}' not found"}

        try:
            # Get basic table info
            result = {
                "row_count": self.tables[table_name]["row_count"],
                "column_count": self.tables[table_name]["column_count"],
                "columns": self.tables[table_name]["columns"],
                "column_types": {},
                "numeric_stats": {},
                "categorical_stats": {},
            }

            # Get a sample of the data to analyze column types
            sample_df = self.execute_query(f"SELECT * FROM {table_name} LIMIT 1000")

            for column in sample_df.columns:
                # Determine column type
                if pd.api.types.is_numeric_dtype(sample_df[column]):
                    result["column_types"][column] = "numeric"

                    # Calculate numeric statistics
                    stats_query = f"""
                    SELECT 
                        MIN({column}) as min_value,
                        MAX({column}) as max_value,
                        AVG({column}) as avg_value,
                        STDDEV({column}) as std_value,
                        COUNT({column}) as count_value,
                        COUNT(DISTINCT {column}) as unique_count
                    FROM {table_name}
                    """

                    try:
                        stats_df = self.execute_query(stats_query)

                        result["numeric_stats"][column] = {
                            "min": float(stats_df["min_value"].iloc[0])
                            if not pd.isna(stats_df["min_value"].iloc[0])
                            else None,
                            "max": float(stats_df["max_value"].iloc[0])
                            if not pd.isna(stats_df["max_value"].iloc[0])
                            else None,
                            "avg": float(stats_df["avg_value"].iloc[0])
                            if not pd.isna(stats_df["avg_value"].iloc[0])
                            else None,
                            "std": float(stats_df["std_value"].iloc[0])
                            if not pd.isna(stats_df["std_value"].iloc[0])
                            else None,
                            "count": int(stats_df["count_value"].iloc[0]),
                            "unique_count": int(stats_df["unique_count"].iloc[0]),
                        }
                    except Exception as e:
                        logger.error(
                            f"Error calculating numeric stats for column {column}: {str(e)}"
                        )
                        result["numeric_stats"][column] = {"error": str(e)}

                else:
                    result["column_types"][column] = "categorical"

                    # Calculate top categories (limited to avoid large query results)
                    top_values_query = f"""
                    SELECT 
                        {column} as value,
                        COUNT(*) as count
                    FROM {table_name}
                    WHERE {column} IS NOT NULL
                    GROUP BY {column}
                    ORDER BY count DESC
                    LIMIT 5
                    """

                    try:
                        top_values_df = self.execute_query(top_values_query)

                        # Convert to list of {value: x, count: y} dictionaries
                        top_values = []
                        for _, row in top_values_df.iterrows():
                            top_values.append(
                                {"value": str(row["value"]), "count": int(row["count"])}
                            )

                        # Get count of unique values
                        unique_count_query = (
                            f"SELECT COUNT(DISTINCT {column}) as unique_count FROM {table_name}"
                        )
                        unique_count_df = self.execute_query(unique_count_query)
                        unique_count = int(unique_count_df["unique_count"].iloc[0])

                        result["categorical_stats"][column] = {
                            "top_values": top_values,
                            "unique_count": unique_count,
                        }
                    except Exception as e:
                        logger.error(
                            f"Error calculating categorical stats for column {column}: {str(e)}"
                        )
                        result["categorical_stats"][column] = {"error": str(e)}

            return result

        except Exception as e:
            logger.error(f"Error getting data statistics for table '{table_name}'", exc_info=True)
            return {"error": str(e)}
