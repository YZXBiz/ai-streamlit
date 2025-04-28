"""
Data source interfaces for loading different types of data files.

This module provides the abstract base class for all data sources that can be loaded
into the system for analysis.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandasai as pai


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, source: str, name: str, description: Optional[str] = None):
        """
        Initialize the data source.

        Args:
            source: The source identifier (file path, connection string, etc.)
            name: Name of the dataset
            description: Optional description of the dataset
        """
        self.source = source
        self.name = name
        self.description = description or f"Data from {source}"

    @abstractmethod
    def load(self) -> pai.DataFrame:
        """
        Load data from the source into a PandasAI DataFrame.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        pass 