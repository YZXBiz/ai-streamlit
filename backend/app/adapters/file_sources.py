"""
Concrete implementations of data sources for various file types.

This module provides classes to handle loading different file formats (CSV, Excel, Parquet)
into pandas DataFrames for analysis.
"""

from typing import Any, Optional, Union

import pandas as pd
import pandasai as pai

from ..ports.datasource import DataSource


class CSVSource(DataSource):
    """Data source for CSV files."""

    def __init__(
        self,
        source: str,
        name: str,
        description: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the CSV data source.

        Args:
            source: Path to the CSV file
            name: Name of the dataset
            description: Optional description of the dataset
            **kwargs: Additional arguments to pass to pd.read_csv
        """
        super().__init__(source, name, description)
        self.kwargs = kwargs

    def load(self) -> pai.DataFrame:
        """
        Load data from the CSV file.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.source, **self.kwargs)

        # Convert to PandasAI DataFrame
        return pai.DataFrame(df, name=self.name, description=self.description)


class ExcelSource(DataSource):
    """Data source for Excel files."""

    def __init__(
        self,
        source: str,
        name: str,
        description: str | None = None,
        sheet_name: str | int | None = 0,
        **kwargs: Any,
    ):
        """
        Initialize the Excel data source.

        Args:
            source: Path to the Excel file
            name: Name of the dataset
            description: Optional description of the dataset
            sheet_name: Name or index of the sheet to load (default: 0, first sheet)
            **kwargs: Additional arguments to pass to pd.read_excel
        """
        super().__init__(source, name, description)
        self.sheet_name = sheet_name
        self.kwargs = kwargs

    def load(self) -> pai.DataFrame:
        """
        Load data from the Excel file.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(self.source, sheet_name=self.sheet_name, **self.kwargs)

        # Convert to PandasAI DataFrame
        return pai.DataFrame(df, name=self.name, description=self.description)


class ParquetSource(DataSource):
    """Data source for Parquet files."""

    def __init__(
        self,
        source: str,
        name: str,
        description: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the Parquet data source.

        Args:
            source: Path to the Parquet file
            name: Name of the dataset
            description: Optional description of the dataset
            **kwargs: Additional arguments to pass to pd.read_parquet
        """
        super().__init__(source, name, description)
        self.kwargs = kwargs

    def load(self) -> pai.DataFrame:
        """
        Load data from the Parquet file.

        Returns:
            pai.DataFrame: A PandasAI DataFrame containing the loaded data
        """
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(self.source, **self.kwargs)

        # Convert to PandasAI DataFrame
        return pai.DataFrame(df, name=self.name, description=self.description)
