"""DuckDB adapter for efficient in-memory analytics."""

import os
from typing import Any

import duckdb
import pandas as pd
import pandasai as pai
from pandasai import DataFrame

from backend.app.ports.datasource import DataSource


class DuckDBManager:
    """
    Manager for DuckDB connections and operations.

    This class provides methods for:
    - Loading data from files into DuckDB
    - Executing SQL queries on the data
    - Converting between DuckDB and pandas DataFrames
    """

    def __init__(self, database_path: str | None = None):
        """
        Initialize the DuckDB manager.

        Args:
            database_path: Optional path to a persistent DuckDB database file.
                           If None, an in-memory database is used.
        """
        # Use in-memory database by default
        self.database_path = database_path or ":memory:"

        # Create a connection
        self.conn = duckdb.connect(self.database_path)

        # Dictionary to track registered tables
        self.tables = {}  # table_name -> metadata

    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        self.close()

    def load_file(
        self, file_path: str, table_name: str, schema: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """
        Load a file into DuckDB and register it as a table.

        Args:
            file_path: Path to the file to load
            table_name: Name to register the table as
            schema: Optional schema definition

        Returns:
            pandas DataFrame with the loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()

        # Create the table
        if file_ext == ".csv":
            # Load CSV file
            if schema:
                # Create table with schema
                schema_str = ", ".join([f"{k} {v}" for k, v in schema.items()])
                self.conn.execute(f"CREATE TABLE {table_name} ({schema_str})")
                self.conn.execute(f"COPY {table_name} FROM '{file_path}' (AUTO_DETECT TRUE)")
            else:
                # Auto-detect schema
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')"
                )
        elif file_ext in (".parquet", ".pq"):
            # Load Parquet file
            self.conn.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM parquet_scan('{file_path}')"
            )
        elif file_ext in (".xlsx", ".xls"):
            # For Excel, we need to use pandas as an intermediate step
            df = pd.read_excel(file_path)
            self.register_dataframe(df, table_name)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Store metadata
        self.tables[table_name] = {
            "source": file_path,
            "type": file_ext[1:],  # Remove the dot
            "row_count": self.get_row_count(table_name),
        }

        # Return as pandas DataFrame
        return self.to_pandas(table_name)

    def register_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Register a pandas DataFrame as a DuckDB table.

        Args:
            df: pandas DataFrame to register
            table_name: Name to register the table as
        """
        # Register the DataFrame
        self.conn.register(table_name, df)

        # Store metadata
        self.tables[table_name] = {
            "source": "pandas",
            "type": "dataframe",
            "row_count": len(df),
        }

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.

        Args:
            query: SQL query to execute

        Returns:
            pandas DataFrame with the query results
        """
        return self.conn.execute(query).fetchdf()

    def to_pandas(self, table_name: str) -> pd.DataFrame:
        """
        Convert a DuckDB table to a pandas DataFrame.

        Args:
            table_name: Name of the table to convert

        Returns:
            pandas DataFrame with the table data
        """
        return self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()

    def to_pandasai(self, table_name: str, description: str = "") -> DataFrame:
        """
        Convert a DuckDB table to a PandasAI DataFrame.

        Args:
            table_name: Name of the table to convert
            description: Optional description for the DataFrame

        Returns:
            PandasAI DataFrame with the table data
        """
        pandas_df = self.to_pandas(table_name)
        return DataFrame(pandas_df, name=table_name, description=description)

    def get_table_names(self) -> list[str]:
        """
        Get the names of all registered tables.

        Returns:
            List of table names
        """
        return list(self.tables.keys())

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        """
        Get the schema of a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary mapping column names to their types
        """
        result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
        return {row["column_name"]: row["column_type"] for _, row in result.iterrows()}

    def get_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows
        """
        return self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    def get_table_preview(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Get a preview of a table.

        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            pandas DataFrame with the preview data
        """
        return self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()

    def get_table_info(self, table_name: str) -> dict[str, Any]:
        """
        Get information about a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        schema = self.get_table_schema(table_name)
        row_count = self.get_row_count(table_name)
        metadata = self.tables.get(table_name, {})

        return {
            "name": table_name,
            "schema": schema,
            "row_count": row_count,
            "source": metadata.get("source", "unknown"),
            "type": metadata.get("type", "unknown"),
        }

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from DuckDB.

        Args:
            table_name: Name of the table to drop
        """
        try:
            # Try to drop as a TABLE
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except duckdb.CatalogException:
            # If that fails, try to drop as a VIEW
            try:
                self.conn.execute(f"DROP VIEW IF EXISTS {table_name}")
            except Exception as e:
                # If both fail, report the error
                print(f"Failed to drop {table_name}: {e}")

        # Remove from tracked tables
        if table_name in self.tables:
            del self.tables[table_name]


class DuckDBDataSource(DataSource):
    """
    DuckDB-backed data source implementation.

    This adapter can be initialized with either:
    1. A file path (DuckDB will load the file)
    2. A pandas DataFrame (DuckDB will register it as a table)
    """

    def __init__(
        self,
        file_path: str = None,
        df: pd.DataFrame = None,
        table_name: str = None,
        description: str = "",
    ):
        """
        Initialize the DuckDB data source.

        Args:
            file_path: Path to a data file (CSV, Parquet, Excel)
            df: Pandas DataFrame to wrap
            table_name: Name to register the table as (optional)
            description: Optional description for the data
        """
        # Determine the data source name/identifier
        source = file_path if file_path else "dataframe"
        name = table_name or (
            os.path.basename(file_path).split(".")[0] if file_path else "data_table"
        )

        # Call parent constructor
        super().__init__(source=source, name=name, description=description)

        # Store local attributes
        self.file_path = file_path
        self.df = df
        self.table_name = name

        # Initialize DuckDB manager
        self.connection = duckdb.connect(":memory:")

        # Load the data
        if file_path:
            self._load_file()
        elif df is not None:
            self._register_dataframe()

    def _load_file(self):
        """Load the file into DuckDB."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Determine file type from extension
        file_ext = os.path.splitext(self.file_path)[1].lower()

        # Create the table
        if file_ext == ".csv":
            self.connection.execute(
                f"CREATE TABLE {self.table_name} AS SELECT * FROM read_csv_auto('{self.file_path}')"
            )
        elif file_ext in (".parquet", ".pq"):
            self.connection.execute(
                f"CREATE TABLE {self.table_name} AS SELECT * FROM parquet_scan('{self.file_path}')"
            )
        elif file_ext in (".xlsx", ".xls"):
            # For Excel, we need to use pandas as an intermediate step
            temp_df = pd.read_excel(self.file_path)
            self.connection.register(self.table_name, temp_df)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _register_dataframe(self):
        """Register the DataFrame with DuckDB."""
        self.connection.register(self.table_name, self.df)

    def load(self) -> pai.DataFrame:
        """
        Load data from the source into a PandasAI DataFrame.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        # Get the data as a pandas DataFrame
        pandas_df = self.get_df()

        # Convert to PandasAI DataFrame
        return pai.DataFrame(pandas_df, name=self.name, description=self.description)

    def get_df(self) -> pd.DataFrame:
        """Get the data as a pandas DataFrame."""
        if self.df is not None:
            return self.df

        # Query from DuckDB
        return self.connection.execute(f"SELECT * FROM {self.table_name}").fetchdf()

    def get_raw_data(self) -> str:
        """Get the data as a raw string."""
        df = self.get_df()
        return df.to_csv(index=False)

    def get_metadata(self) -> dict:
        """Get metadata about the data source."""
        schema = self.get_schema()

        metadata = {
            "table_name": self.table_name,
            "description": self.description,
            "row_count": len(self.get_df()),
            "column_count": len(schema["columns"]) if schema else 0,
        }

        if self.file_path:
            metadata["file_path"] = self.file_path
            metadata["file_size"] = os.path.getsize(self.file_path)
            metadata["file_type"] = os.path.splitext(self.file_path)[1].lower()[1:]

        return metadata

    def get_schema(self) -> dict:
        """Get the schema of the data."""
        result = self.connection.execute(f"DESCRIBE {self.table_name}").fetchdf()

        columns = []
        for _, row in result.iterrows():
            columns.append({"name": row["column_name"], "type": row["column_type"]})

        return {"columns": columns}

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query on the data."""
        return self.connection.execute(query).fetchdf()
