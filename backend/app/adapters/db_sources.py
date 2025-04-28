"""
Database source implementations for the data source interface.

This module provides classes to handle loading data from database connections
into pandas DataFrames for analysis.
"""

from typing import Optional

import pandas as pd
import pandasai as pai
from sqlalchemy import create_engine, text

from ..ports.datasource import DataSource


class SQLSource(DataSource):
    """Data source for SQL database connections."""

    def __init__(
        self,
        connection_string: str,
        query: str,
        name: str,
        description: str | None = None,
    ):
        """
        Initialize the SQL data source.

        Args:
            connection_string: SQLAlchemy connection string
            query: SQL query to execute
            name: Name of the dataset
            description: Optional description of the dataset
        """
        super().__init__(connection_string, name, description)
        self.query = query

    def load(self) -> pai.DataFrame:
        """
        Load data from the SQL query.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        # Create a SQLAlchemy engine
        engine = create_engine(self.source)

        # Read the query results into a pandas DataFrame
        with engine.connect() as connection:
            df = pd.read_sql(text(self.query), connection)

        # Convert to PandasAI DataFrame
        return pai.DataFrame(df, name=self.name, description=self.description)
