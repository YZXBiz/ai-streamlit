"""Base classes for data writers."""

from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import pydantic as pdt

from clustering.utils import ensure_directory


class Writer(pdt.BaseModel, ABC):
    """Base class for data writers."""

    @abstractmethod
    def write(self, data: pl.DataFrame) -> None:
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
