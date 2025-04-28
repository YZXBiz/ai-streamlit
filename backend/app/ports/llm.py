"""Interface for LLM-powered data analysis services."""

from abc import ABC, abstractmethod
from typing import Any

from pandasai import DataFrame


class DataAnalysisService(ABC):
    """
    Interface for a service that provides LLM-powered data analysis.

    This abstraction allows the application to use different implementations
    of data analysis services, such as PandasAI, without tight coupling.
    """

    @abstractmethod
    async def load_dataframe(
        self, file_path: str, name: str, description: str = ""
    ) -> DataFrame | Any:
        """
        Load a dataframe from a file.

        Args:
            file_path: Path to the file to load
            name: Name for the dataframe
            description: Optional description

        Returns:
            The loaded dataframe
        """
        pass

    @abstractmethod
    async def query_dataframe(self, query: str, dataframe_name: str) -> Any:
        """
        Query a dataframe using natural language.

        Args:
            query: Natural language query
            dataframe_name: Name of the dataframe to query

        Returns:
            Query result (can be text, charts, or dataframes)
        """
        pass

    @abstractmethod
    async def create_collection(
        self, dataframe_names: list[str], collection_name: str, description: str = ""
    ) -> Any:
        """
        Create a collection of dataframes for cross-dataframe analysis.

        Args:
            dataframe_names: Names of dataframes to include in collection
            collection_name: Name for the collection
            description: Optional description

        Returns:
            The created collection
        """
        pass

    @abstractmethod
    async def query_collection(self, query: str, collection_name: str) -> Any:
        """
        Query a collection of dataframes using natural language.

        Args:
            query: Natural language query
            collection_name: Name of the collection to query

        Returns:
            Query result (can be text, charts, or dataframes)
        """
        pass

    @abstractmethod
    async def get_dataframe_info(self, dataframe_name: str) -> dict:
        """
        Get information about a dataframe.

        Args:
            dataframe_name: Name of the dataframe

        Returns:
            Dictionary with dataframe information
        """
        pass
