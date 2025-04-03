"""Base classes for data readers."""

from abc import ABC, abstractmethod
from typing import Optional

import polars as pl
import pydantic as pdt


class Reader(pdt.BaseModel, ABC):
    """Base class for data readers."""

    limit: Optional[int] = None

    @abstractmethod
    def read(self) -> pl.DataFrame:
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
