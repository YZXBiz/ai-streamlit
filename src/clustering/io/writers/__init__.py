"""Data writers for the clustering pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pydantic as pdt

from clustering.utils import ensure_directory


class Writer(pdt.BaseModel, ABC):
    """Base class for data writers."""

    @abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write data to destination.

        Args:
            data: DataFrame to write
        """


class FileWriter(Writer):
    """Base class for file-based writers."""

    path: str
    create_parent_dirs: bool = True

    def __str__(self) -> str:
        """Return string representation of the writer."""
        return f"{self.__class__.__name__}(path={self.path})"

    def _prepare_path(self) -> None:
        """Prepare the path by creating parent directories if needed."""
        if self.create_parent_dirs:
            ensure_directory(Path(self.path).parent)


class ParquetWriter(FileWriter):
    """Writer for Parquet files."""

    compression: str = "snappy"

    def write(self, data: pd.DataFrame) -> None:
        """Write data to Parquet file.

        Args:
            data: DataFrame to write
        """
        self._prepare_path()
        data.to_parquet(self.path, compression=self.compression)


class CSVWriter(FileWriter):
    """Writer for CSV files."""

    delimiter: str = ","
    index: bool = False

    def write(self, data: pd.DataFrame) -> None:
        """Write data to CSV file.

        Args:
            data: DataFrame to write
        """
        self._prepare_path()
        data.to_csv(
            self.path,
            delimiter=self.delimiter,
            index=self.index,
        )


# Export the writer classes
__all__ = [
    "Writer",
    "FileWriter",
    "ParquetWriter",
    "CSVWriter",
]
