"""
PandasAI LLM service adapter.

This module provides the PandasAI integration for data analysis services.
"""

import os
from typing import Any, Optional

import pandas as pd
import pandasai as pai
from pandasai import Agent, DataFrame
from pandasai.config import Config

from ..adapters.db_duckdb import DuckDBManager
from ..adapters.sandbox import CodeSandbox
from ..core.config import settings
from ..ports.llm import DataAnalysisService


class PandasAiAdapter(DataAnalysisService):
    """PandasAI implementation of the DataAnalysisService interface."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the PandasAI adapter.

        Args:
            api_key: Optional API key for PandasAI. If not provided,
                    it will use the value from environment variables.
        """
        # Set the API key in environment variables if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Initialize dataframe storage
        self.dataframes: dict[str, DataFrame] = {}
        # Use a simple dictionary for dataframe management
        self.manager = {"dataframes": {}, "descriptions": {}, "collections": {}}

        # Initialize DuckDB for efficient analytics
        self.duckdb = DuckDBManager()

        # Initialize the PandasAI Agent with configuration
        config = Config(
            display="streamlit",
            save_charts=True,
            save_charts_path=settings.CHARTS_DIR,
            enforce_privacy=settings.ENFORCE_PRIVACY,
            enable_cache=settings.ENABLE_CACHE,
            max_retries=settings.MAX_RETRIES,
            use_error_correction_framework=True,
        )

        # Create a sandbox for secure code execution
        self.sandbox = CodeSandbox()

        # Initialize the PandasAI Agent with sandbox
        self.agent = Agent(
            config=config,
            memory_size=settings.MEMORY_SIZE,
            enable_cache=settings.ENABLE_CACHE,
        )

        # Create a temporary directory for chart storage
        os.makedirs(settings.CHARTS_DIR, exist_ok=True)

        self.last_result: Any = None
        self.last_code: str | None = None
        self.last_error: str | None = None

    async def execute_code_safely(
        self, code: str, dataframe: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """
        Execute code in a sandbox for security.

        Args:
            code: Python code to execute
            dataframe: Optional pandas DataFrame to make available to the code

        Returns:
            Dictionary with execution results
        """
        # Add the dataframe to the code if provided
        if dataframe is not None:
            # Prepend code to make the dataframe available
            code = f"import pandas as pd\ndf = pd.DataFrame({dataframe.to_dict()})\n\n{code}"

        # Execute the code in the sandbox
        result = self.sandbox.execute_code(code)

        # Store the code for reference
        self.last_code = code

        # Store error if any
        if not result["success"]:
            self.last_error = result["error"]

        return result

    async def load_dataframe(
        self, file_path: str, name: str, description: str = ""
    ) -> DataFrame | Any:
        """
        Load a dataframe from a file.

        Args:
            file_path: Path to the file
            name: Name to assign to the dataframe
            description: Optional description of the dataframe

        Returns:
            The loaded DataFrame
        """
        # Determine file type based on extension
        if file_path.endswith(".csv"):
            return await self._load_csv(file_path, name, description)
        elif file_path.endswith((".xls", ".xlsx")):
            return await self._load_excel(file_path, name, description)
        elif file_path.endswith((".parquet", ".pq")):
            return await self._load_parquet(file_path, name, description)
        else:
            raise ValueError(f"Unsupported file format for file: {file_path}")

    async def _load_csv(self, file_path: str, name: str, description: str = "") -> DataFrame:
        """Load a CSV file into a PandasAI DataFrame."""
        # Use pandas to read the file
        pandas_df = pd.read_csv(file_path)

        # Convert to PandasAI DataFrame
        df = DataFrame(pandas_df)

        # Store the dataframe
        self.dataframes[name] = df

        # Add the dataframe to the agent
        self.agent.add_data(df, name=name)

        # Register with the manager
        self.manager["dataframes"][name] = df
        self.manager["descriptions"][name] = description

        return df

    async def _load_excel(
        self, file_path: str, name: str, sheet_name: str | None = None, description: str = ""
    ) -> DataFrame:
        """Load an Excel file into a PandasAI DataFrame."""
        # Use pandas to read the file
        pandas_df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Convert to PandasAI DataFrame
        df = DataFrame(pandas_df)

        # Store the dataframe
        self.dataframes[name] = df

        # Add the dataframe to the agent
        self.agent.add_data(df, name=name)

        # Register with the manager
        self.manager["dataframes"][name] = df
        self.manager["descriptions"][name] = description

        return df

    async def _load_parquet(self, file_path: str, name: str, description: str = "") -> DataFrame:
        """Load a Parquet file into a PandasAI DataFrame."""
        # Use pandas to read the file
        pandas_df = pd.read_parquet(file_path)

        # Convert to PandasAI DataFrame
        df = DataFrame(pandas_df)

        # Store the dataframe
        self.dataframes[name] = df

        # Add the dataframe to the agent
        self.agent.add_data(df, name=name)

        # Register with the manager
        self.manager["dataframes"][name] = df
        self.manager["descriptions"][name] = description

        return df

    async def query_dataframe(self, query: str, dataframe_name: str) -> Any:
        """
        Query a dataframe using natural language.

        Args:
            query: The natural language query
            dataframe_name: Name of the dataframe to query

        Returns:
            Query results
        """
        # Verify dataframe exists
        if dataframe_name not in self.dataframes:
            raise ValueError(f"Dataframe '{dataframe_name}' not found")

        try:
            # Get the dataframe
            df = self.dataframes[dataframe_name]

            # Load into DuckDB for efficient querying if not already loaded
            if dataframe_name not in self.duckdb.tables:
                self.duckdb.register_dataframe(df._obj, dataframe_name)

            # Get schema information for context
            schema_info = self.duckdb.get_table_schema(dataframe_name)
            schema_str = ", ".join([f"{col}: {dtype}" for col, dtype in schema_info.items()])

            # Create a context with dataframe information
            context = (
                f"Analyzing dataframe '{dataframe_name}' with schema: {schema_str}.\n"
                f"The dataframe has {self.duckdb.get_row_count(dataframe_name)} rows.\n\n"
            )

            # Run the query through the agent with privacy and sandbox
            augmented_prompt = f"{context}{query}"
            result = self.agent.chat(augmented_prompt)

            # Store the result for later reference
            self.last_result = result

            # Store code information if available
            try:
                if hasattr(result, "code"):
                    self.last_code = result.code

                    # Validate the code in our sandbox
                    await self.execute_code_safely(result.code, df._obj)
                else:
                    self.last_code = None
            except Exception as code_error:
                self.last_error = str(code_error)
                self.last_code = None

            return result
        except Exception as e:
            # Log the error and provide a user-friendly message
            error_msg = str(e)
            self.last_error = error_msg

            if "Result must be in the format of dictionary of type and value" in error_msg:
                raise ValueError(
                    "The PandasAI model returned an invalid response format. "
                    "This usually happens due to version incompatibility issues."
                ) from e
            raise ValueError(f"PandasAI query failed: {error_msg}") from e

    async def create_collection(
        self, dataframe_names: list[str], collection_name: str, description: str = ""
    ) -> Any:
        """
        Create a collection of dataframes for cross-dataframe analysis.

        Args:
            dataframe_names: List of dataframe names to include
            collection_name: Name for the collection
            description: Optional description of the collection

        Returns:
            The created collection
        """
        # Get the dataframes
        dfs = []
        for name in dataframe_names:
            if name not in self.dataframes:
                raise ValueError(f"Dataframe '{name}' not found")
            dfs.append(self.dataframes[name])

        # Create a collection
        collection = pai.Collection(dfs, name=collection_name, description=description)

        # Store in the manager
        self.manager["collections"][collection_name] = collection

        return collection

    async def query_collection(self, query: str, collection_name: str) -> Any:
        """
        Query a collection of dataframes using natural language.

        Args:
            query: The natural language query
            collection_name: Name of the collection to query

        Returns:
            Query results
        """
        # Get the collection from the manager
        if collection_name not in self.manager["collections"]:
            raise ValueError(f"Collection '{collection_name}' not found")

        collection = self.manager["collections"][collection_name]

        try:
            # Create context with collection information
            context = (
                f"Analyzing collection '{collection_name}' with "
                f"{len(collection.dataframes)} dataframes."
            )

            # Full prompt with context and query
            full_prompt = f"{context}\n\nQuestion: {query}"

            return collection.chat(full_prompt)
        except Exception as e:
            raise ValueError(f"Error querying collection: {str(e)}") from e

    async def get_dataframe_info(self, dataframe_name: str) -> dict:
        """
        Get information about a dataframe.

        Args:
            dataframe_name: Name of the dataframe

        Returns:
            Dictionary with dataframe information
        """
        # Get the dataframe from the manager
        if dataframe_name not in self.dataframes:
            raise ValueError(f"Dataframe '{dataframe_name}' not found")

        df = self.dataframes[dataframe_name]

        # Access the underlying pandas DataFrame
        pandas_df = df._obj

        # Get basic information
        info = {
            "name": dataframe_name,
            "description": self.manager["descriptions"].get(dataframe_name, ""),
            "shape": {
                "rows": pandas_df.shape[0],
                "columns": pandas_df.shape[1],
            },
            "columns": [],
            "memory_usage": pandas_df.memory_usage(deep=True).sum(),
        }

        # Column information
        for col in pandas_df.columns:
            col_info = {
                "name": col,
                "dtype": str(pandas_df[col].dtype),
                "nullable": pandas_df[col].isna().any(),
                "unique_values": pandas_df[col].nunique(),
            }

            # Sample values (up to 5)
            sample_values = pandas_df[col].dropna().head(5).tolist()
            col_info["sample_values"] = sample_values

            info["columns"].append(col_info)

        return info
