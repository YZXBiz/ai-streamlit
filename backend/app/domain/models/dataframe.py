"""
DataFrame domain models.

This module defines the core domain models for working with dataframes.
"""

from collections.abc import Sequence

import pandas as pd


class DataFrameCollection:
    """
    A collection of pandas DataFrames for cross-dataframe analysis.

    This class provides methods for organizing multiple dataframes together
    and treating them as a logical group.
    """

    def __init__(
        self,
        dataframes: Sequence[pd.DataFrame],
        name: str,
        dataframe_names: list[str] | None = None,
        description: str = "",
    ):
        """
        Initialize a collection of dataframes.

        Args:
            dataframes: Sequence of pandas DataFrames to include in the collection
            name: Name of the collection
            dataframe_names: List of names corresponding to the dataframes
            description: Optional description of the collection
        """
        self.dataframes = list(dataframes)
        self.name = name
        self.description = description

        # Generate default names if not provided
        if dataframe_names is None:
            self.dataframe_names = [f"dataframe_{i}" for i in range(len(dataframes))]
        # Handle case where fewer names than dataframes are provided
        elif len(dataframe_names) < len(dataframes):
            self.dataframe_names = list(dataframe_names)
            for i in range(len(dataframe_names), len(dataframes)):
                self.dataframe_names.append(f"dataframe_{i}")
        else:
            self.dataframe_names = list(dataframe_names)

    def __len__(self) -> int:
        """Return the number of dataframes in the collection."""
        return len(self.dataframes)

    def add_dataframe(self, dataframe: pd.DataFrame, name: str | None = None) -> bool:
        """
        Add a dataframe to the collection.

        Args:
            dataframe: pandas DataFrame to add
            name: Name of the dataframe

        Returns:
            True if the dataframe was added, False if the name already exists
        """
        # Generate default name if not provided
        if name is None:
            name = f"dataframe_{len(self.dataframes)}"

        # Check for duplicate name
        if name in self.dataframe_names:
            return False

        # Add to our internal tracking
        self.dataframes.append(dataframe)
        self.dataframe_names.append(name)

        return True

    def remove_dataframe(self, name: str) -> bool:
        """
        Remove a dataframe from the collection.

        Args:
            name: Name of the dataframe to remove

        Returns:
            True if the dataframe was removed, False if it wasn't found
        """
        if name in self.dataframe_names:
            index = self.dataframe_names.index(name)

            # Remove from our internal tracking
            self.dataframe_names.pop(index)
            self.dataframes.pop(index)

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
        if name in self.dataframe_names:
            index = self.dataframe_names.index(name)
            return self.dataframes[index]
        return None

    def get_dataframes(self) -> list[pd.DataFrame]:
        """
        Get all dataframes in the collection.

        Returns:
            List of pandas DataFrames
        """
        return self.dataframes

    def get_query_context(self) -> str:
        """
        Get context information for this collection.

        Returns:
            String with context information including dataframe names
        """
        context = f"Collection '{self.name}'"
        if self.description:
            context += f": {self.description}"
        context += (
            f"\nContains {len(self.dataframes)} dataframes: {', '.join(self.dataframe_names)}"
        )
        return context
