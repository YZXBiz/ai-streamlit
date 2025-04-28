"""
AnalyzerService for data analysis operations.

This service provides a high-level interface for data loading, analysis, and visualization.
"""

import tempfile

import pandas as pd

from ..domain.models.dataframe import DataFrameCollection
from ..services.dataframe_service import DataFrameService


class AnalyzerService:
    """
    Main service for data analysis operations.

    This service provides a high-level interface for loading, querying,
    and analyzing data using pandas DataFrames.
    """

    def __init__(self):
        """Initialize the analyzer service."""
        self.dataframe_service = DataFrameService()
        self.charts_dir = tempfile.mkdtemp(prefix="pandas_charts_")

    # DataFrame loading operations

    def load_csv(self, file_path: str, name: str, description: str = "") -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Args:
            file_path: Path to the CSV file
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded DataFrame
        """
        return self.dataframe_service.load_csv(file_path, name, description)

    def load_excel(
        self, file_path: str, name: str, sheet_name: str | None = None, description: str = ""
    ) -> pd.DataFrame:
        """
        Load an Excel file into a DataFrame.

        Args:
            file_path: Path to the Excel file
            name: Name to assign to the dataframe
            sheet_name: Optional sheet name to load
            description: Optional description for the dataframe

        Returns:
            The loaded DataFrame
        """
        return self.dataframe_service.load_excel(file_path, name, sheet_name, description)

    def load_parquet(self, file_path: str, name: str, description: str = "") -> pd.DataFrame:
        """
        Load a Parquet file into a DataFrame.

        Args:
            file_path: Path to the Parquet file
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded DataFrame
        """
        return self.dataframe_service.load_parquet(file_path, name, description)

    def load_dataframe(self, df: pd.DataFrame, name: str, description: str = "") -> pd.DataFrame:
        """
        Register an existing DataFrame.

        Args:
            df: Pandas DataFrame to register
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The registered DataFrame
        """
        self.dataframe_service.register_dataframe(df, name, description)
        return df

    def load_sql(
        self, connection_string: str, query: str, name: str, description: str = ""
    ) -> pd.DataFrame:
        """
        Load data from SQL into a DataFrame.

        Args:
            connection_string: SQLAlchemy connection string
            query: SQL query to execute
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded DataFrame
        """
        return self.dataframe_service.load_sql(connection_string, query, name, description)

    # DataFrame retrieval and information operations

    def get_dataframe(self, name: str) -> pd.DataFrame | None:
        """
        Get a DataFrame by name.

        Args:
            name: Name of the dataframe to retrieve

        Returns:
            The DataFrame if found, None otherwise
        """
        return self.dataframe_service.get_dataframe(name)

    def get_dataframe_preview(self, name: str, rows: int = 5) -> pd.DataFrame | None:
        """
        Get a preview of a DataFrame.

        Args:
            name: Name of the dataframe
            rows: Number of rows to preview

        Returns:
            A preview of the dataframe, or None if not found
        """
        return self.dataframe_service.get_dataframe_preview(name, rows)

    def get_dataframe_schema(self, name: str) -> dict | None:
        """
        Get the schema of a DataFrame.

        Args:
            name: Name of the dataframe

        Returns:
            Dictionary with schema information, or None if not found
        """
        return self.dataframe_service.get_dataframe_schema(name)

    def get_dataframe_stats(self, name: str) -> pd.DataFrame | None:
        """
        Get statistics for a DataFrame.

        Args:
            name: Name of the dataframe

        Returns:
            DataFrame with statistics, or None if not found
        """
        return self.dataframe_service.get_dataframe_stats(name)

    def get_all_dataframes(self) -> list[tuple[str, pd.DataFrame]]:
        """
        Get all registered DataFrames.

        Returns:
            List of tuples containing dataframe names and objects
        """
        return self.dataframe_service.get_all_dataframes()

    def get_dataframe_names(self) -> list[str]:
        """
        Get names of all registered DataFrames.

        Returns:
            List of dataframe names
        """
        return self.dataframe_service.get_dataframe_names()

    # Collection operations

    def create_collection(
        self, dataframe_names: list[str], collection_name: str, description: str = ""
    ) -> DataFrameCollection:
        """
        Create a collection of dataframes.

        Args:
            dataframe_names: List of dataframe names to include in the collection
            collection_name: Name for the collection
            description: Optional description for the collection

        Returns:
            The created collection

        Raises:
            ValueError: If any of the specified dataframe names don't exist
        """
        return self.dataframe_service.create_collection(
            dataframe_names, collection_name, description
        )

    def get_collection(self, name: str) -> DataFrameCollection | None:
        """
        Get a collection by name.

        Args:
            name: Name of the collection to retrieve

        Returns:
            The collection if found, None otherwise
        """
        return self.dataframe_service.get_collection(name)

    def get_collection_names(self) -> list[str]:
        """
        Get names of all collections.

        Returns:
            List of collection names
        """
        return self.dataframe_service.get_collection_names()

    # Relationship operations

    def add_relationship(
        self, source_df: str, source_column: str, target_df: str, target_column: str
    ) -> bool:
        """
        Add a relationship between dataframes.

        Args:
            source_df: Name of the source dataframe
            source_column: Column in the source dataframe
            target_df: Name of the target dataframe
            target_column: Column in the target dataframe

        Returns:
            True if the relationship was added successfully, False otherwise
        """
        return self.dataframe_service.add_relationship(
            source_df, source_column, target_df, target_column
        )

    def clear_dataframes(self) -> None:
        """
        Clear all dataframes.
        """
        self.dataframe_service.clear()
