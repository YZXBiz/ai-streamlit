"""Data readers for the clustering pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pydantic as pdt


class Reader(pdt.BaseModel, ABC):
    """Base class for data readers."""

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read data from source.

        Returns:
            DataFrame containing the data
        """


class FileReader(Reader):
    """Base class for file-based readers."""

    path: str

    def __str__(self) -> str:
        """Return string representation of the reader."""
        return f"{self.__class__.__name__}(path={self.path})"


class ParquetReader(FileReader):
    """Reader for Parquet files."""

    def read(self) -> pd.DataFrame:
        """Read data from Parquet file.

        Returns:
            DataFrame containing the data
        """
        return pd.read_parquet(self.path)


class CSVReader(FileReader):
    """Reader for CSV files."""

    delimiter: str = ","
    header: int = 0
    index_col: Optional[int] = None

    def read(self) -> pd.DataFrame:
        """Read data from CSV file.

        Returns:
            DataFrame containing the data
        """
        return pd.read_csv(
            self.path,
            delimiter=self.delimiter,
            header=self.header,
            index_col=self.index_col,
        )


# Export the reader classes
__all__ = [
    "Reader",
    "FileReader",
    "ParquetReader",
    "CSVReader",
]
