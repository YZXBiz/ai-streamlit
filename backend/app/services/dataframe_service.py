"""
DataFrameService for managing and analyzing pandas DataFrames.

This service manages dataframes, collections, and relationships between dataframes.
"""

import tempfile

import pandas as pd
import sqlalchemy

from backend.app.domain.models.dataframe import DataFrameCollection


class DataFrameService:
    """
    Service for managing and analyzing pandas DataFrames.

    This service handles registration, organization, and basic analytics
    for pandas DataFrames.
    """

    def __init__(self):
        """Initialize the DataFrame service."""
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.collections: dict[str, DataFrameCollection] = {}
        self.descriptions: dict[str, str] = {}
        self.relationships: set[tuple[str, str, str, str]] = set()
        self.charts_dir = tempfile.mkdtemp(prefix="pandas_charts_")

    def register_dataframe(self, df: pd.DataFrame, name: str, description: str = "") -> bool:
        """
        Register a dataframe with the service.

        Args:
            df: pandas DataFrame to register
            name: Name to assign to the dataframe
            description: Optional description of the dataframe

        Returns:
            True if registration succeeded, False if the name already exists
        """
        if name in self.dataframes:
            return False

        self.dataframes[name] = df
        self.descriptions[name] = description
        return True

    def unregister_dataframe(self, name: str) -> bool:
        """
        Remove a dataframe from the service.

        Args:
            name: Name of the dataframe to remove

        Returns:
            True if the dataframe was removed, False if it wasn't found
        """
        if name in self.dataframes:
            del self.dataframes[name]

            if name in self.descriptions:
                del self.descriptions[name]

            # Remove from collections containing this dataframe
            for collection_name, collection in list(self.collections.items()):
                if name in collection.dataframe_names:
                    # If it's the only dataframe, remove the collection
                    if len(collection.dataframe_names) == 1:
                        del self.collections[collection_name]
                    else:
                        # Otherwise remove just this dataframe from the collection
                        collection.remove_dataframe(name)

            # Remove relationships involving this dataframe
            self.relationships = {
                rel for rel in self.relationships if rel[0] != name and rel[2] != name
            }

            return True
        return False

    def get_dataframe(self, name: str) -> pd.DataFrame | None:
        """
        Get a dataframe by name.

        Args:
            name: Name of the dataframe to retrieve

        Returns:
            The pandas DataFrame if found, None otherwise
        """
        return self.dataframes.get(name)

    def get_all_dataframes(self) -> list[tuple[str, pd.DataFrame]]:
        """
        Get all registered dataframes with their names.

        Returns:
            List of tuples containing (name, dataframe)
        """
        return [(name, df) for name, df in self.dataframes.items()]

    def get_dataframe_names(self) -> list[str]:
        """
        Get the names of all registered dataframes.

        Returns:
            List of dataframe names
        """
        return list(self.dataframes.keys())

    def get_dataframe_description(self, name: str) -> str | None:
        """
        Get the description of a dataframe.

        Args:
            name: Name of the dataframe

        Returns:
            The description if found, None otherwise
        """
        return self.descriptions.get(name)

    def create_collection(
        self, dataframe_names: list[str], collection_name: str, description: str = ""
    ) -> DataFrameCollection:
        """
        Create a collection of dataframes for cross-dataframe analysis.

        Args:
            dataframe_names: List of dataframe names to include in the collection
            collection_name: Name for the new collection
            description: Optional description of the collection

        Returns:
            The created DataFrameCollection

        Raises:
            ValueError: If any of the specified dataframe names don't exist
        """
        dfs = []
        valid_names = []

        for name in dataframe_names:
            if name in self.dataframes:
                dfs.append(self.dataframes[name])
                valid_names.append(name)
            else:
                raise ValueError(f"Dataframe '{name}' not found")

        collection = DataFrameCollection(
            dfs, name=collection_name, dataframe_names=valid_names, description=description
        )
        self.collections[collection_name] = collection
        return collection

    def get_collection(self, name: str) -> DataFrameCollection | None:
        """
        Get a collection by name.

        Args:
            name: Name of the collection to retrieve

        Returns:
            The DataFrameCollection if found, None otherwise
        """
        return self.collections.get(name)

    def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections.

        Returns:
            List of collection names
        """
        return list(self.collections.keys())

    def add_relationship(
        self, source_df: str, source_column: str, target_df: str, target_column: str
    ) -> bool:
        """
        Define a relationship between two dataframes.

        Args:
            source_df: Name of the source dataframe
            source_column: Column name in the source dataframe
            target_df: Name of the target dataframe
            target_column: Column name in the target dataframe

        Returns:
            True if the relationship was added, False if either dataframe doesn't exist
        """
        if source_df not in self.dataframes or target_df not in self.dataframes:
            return False

        # Store the relationship as a tuple
        self.relationships.add((source_df, source_column, target_df, target_column))
        return True

    def load_csv(self, file_path: str, name: str, description: str = "") -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame and register it.

        Args:
            file_path: Path to the CSV file
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded pandas DataFrame
        """
        df = pd.read_csv(file_path)
        self.register_dataframe(df, name, description)
        return df

    def load_excel(
        self, file_path: str, name: str, sheet_name: str | None = None, description: str = ""
    ) -> pd.DataFrame:
        """
        Load an Excel file into a pandas DataFrame and register it.

        Args:
            file_path: Path to the Excel file
            name: Name to assign to the dataframe
            sheet_name: Optional sheet name to load
            description: Optional description for the dataframe

        Returns:
            The loaded pandas DataFrame
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.register_dataframe(df, name, description)
        return df

    def load_parquet(self, file_path: str, name: str, description: str = "") -> pd.DataFrame:
        """
        Load a Parquet file into a pandas DataFrame and register it.

        Args:
            file_path: Path to the Parquet file
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded pandas DataFrame
        """
        df = pd.read_parquet(file_path)
        self.register_dataframe(df, name, description)
        return df

    def load_sql(
        self, connection_string: str, query: str, name: str, description: str = ""
    ) -> pd.DataFrame:
        """
        Load data from a SQL database into a pandas DataFrame and register it.

        Args:
            connection_string: SQLAlchemy connection string
            query: SQL query to execute
            name: Name to assign to the dataframe
            description: Optional description for the dataframe

        Returns:
            The loaded pandas DataFrame
        """
        engine = sqlalchemy.create_engine(connection_string)
        try:
            df = pd.read_sql(query, engine)
            self.register_dataframe(df, name, description)
            return df
        finally:
            engine.dispose()

    def get_dataframe_preview(self, name: str, rows: int = 5) -> pd.DataFrame | None:
        """
        Get a preview of a dataframe.

        Args:
            name: Name of the dataframe
            rows: Number of rows to preview

        Returns:
            A preview of the dataframe, or None if not found
        """
        df = self.get_dataframe(name)
        if df is not None:
            return df.head(rows)
        return None

    def get_dataframe_schema(self, name: str) -> dict | None:
        """
        Get the schema of a dataframe.

        Args:
            name: Name of the dataframe

        Returns:
            Dictionary with schema information, or None if not found
        """
        df = self.get_dataframe(name)
        if df is not None:
            schema = {
                "name": name,
                "description": self.descriptions.get(name, ""),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            return schema
        return None

    def get_dataframe_stats(self, name: str) -> pd.DataFrame | None:
        """
        Get statistics for a dataframe.

        Args:
            name: Name of the dataframe

        Returns:
            DataFrame with statistics, or None if not found
        """
        df = self.get_dataframe(name)
        if df is not None:
            return df.describe()
        return None

    def clear(self) -> None:
        """
        Clear all dataframes, descriptions, relationships, and collections.
        """
        self.dataframes.clear()
        self.descriptions.clear()
        self.relationships.clear()
        self.collections.clear()
