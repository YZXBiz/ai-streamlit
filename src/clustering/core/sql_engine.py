"""SQL engine for DuckDB-based transformations.

This module provides a declarative approach for SQL transformations using DuckDB.
The SQL class represents a SQL query with bindings, allowing for composition
of SQL transformations in a functional way.
"""

from dataclasses import dataclass
from string import Template
from typing import Any, Dict, Literal, Union

import duckdb
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class SQL:
    """Represents a SQL query with optional bindings.

    This class allows for creating SQL statements that can reference other SQL
    objects or dataframes, enabling composition of transformations.

    Args:
        sql: The SQL query as a string
        bindings: Dictionary of variables to bind to the query
    """

    sql: str
    bindings: Dict[str, Any] = None

    def __post_init__(self):
        if self.bindings is None:
            object.__setattr__(self, "bindings", {})


class DuckDB:
    """Wrapper around DuckDB for executing SQL transformations."""

    def __init__(self, options: str = ""):
        """Initialize DuckDB with optional configurations.

        Args:
            options: Configuration options for DuckDB
        """
        self.connection = duckdb.connect(database=":memory:")

    def close(self):
        """Close the DuckDB connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def sql_to_string(self, sql_obj: SQL) -> str:
        """Convert SQL object to string, recursively handling bindings.

        Args:
            sql_obj: SQL object to convert

        Returns:
            Resolved SQL string
        """
        replacements = {}
        for key, value in sql_obj.bindings.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pl.DataFrame):
                # For dataframes, we use a unique identifier
                replacements[key] = f"df_{id(value)}"
            elif isinstance(value, SQL):
                # For SQL objects, recursively convert
                replacements[key] = f"({self.sql_to_string(value)})"
            elif isinstance(value, str):
                # String literals get quoted
                replacements[key] = f"'{value}'"
            elif value is None:
                replacements[key] = "NULL"
            else:
                # Numbers and other basic types
                replacements[key] = str(value)

        return Template(sql_obj.sql).safe_substitute(replacements)

    def collect_dataframes(self, sql_obj: SQL) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
        """Collect all dataframes from SQL object's bindings.

        Args:
            sql_obj: SQL object to collect dataframes from

        Returns:
            Dictionary of dataframe IDs to dataframe objects
        """
        dataframes = {}
        for _, value in sql_obj.bindings.items():
            if isinstance(value, (pd.DataFrame, pl.DataFrame)):
                dataframes[f"df_{id(value)}"] = value
            elif isinstance(value, SQL):
                # Recursively collect dataframes from nested SQL
                dataframes.update(self.collect_dataframes(value))
        return dataframes

    def query(
        self, sql_obj: SQL, output_format: Literal["polars", "pandas", "raw"] = "polars"
    ) -> Union[pl.DataFrame, pd.DataFrame, duckdb.DuckDBPyResult]:
        """Execute a SQL query against registered dataframes.

        Args:
            sql_obj: SQL object to execute
            output_format: Format for the result ("polars", "pandas", or "raw" for DuckDBPyResult)

        Returns:
            Query result in the requested format
        """
        # Convert SQL to string
        sql_str = self.sql_to_string(sql_obj)

        # Collect and register dataframes
        dataframes = self.collect_dataframes(sql_obj)
        for name, df in dataframes.items():
            self.connection.register(name, df)

        # Execute the query
        result = self.connection.execute(sql_str)

        # Return in requested format
        if output_format == "polars":
            return result.pl()
        elif output_format == "pandas":
            return result.df()
        else:
            return result
