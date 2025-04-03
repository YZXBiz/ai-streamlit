"""Base classes for data readers."""

from abc import ABC, abstractmethod
from typing import Optional

import polars as pl
import pydantic as pdt


class Reader(pdt.BaseModel, ABC):
    """Base class for data readers."""

    limit: Optional[int] = None

    def read(self) -> pl.DataFrame:
        """Template method defining the reading algorithm.

        The template method pattern defines the skeleton of the reading algorithm,
        deferring some steps to subclasses. This ensures a consistent reading
        process while allowing specific implementations to vary.

        Returns:
            DataFrame containing the data
        """
        # Step 1: Validate the source
        self._validate_source()

        # Step 2: Read data from source (implemented by subclasses)
        data = self._read_from_source()

        # Step 3: Apply limit if specified
        if self.limit is not None:
            data = data.head(self.limit)

        # Step 4: Post-process the data
        return self._post_process(data)

    def _validate_source(self) -> None:
        """Validate the data source before reading.

        This hook method can be overridden by subclasses to implement
        specific validation logic.

        Raises:
            ValueError: If the source is invalid
        """
        pass

    @abstractmethod
    def _read_from_source(self) -> pl.DataFrame:
        """Read data from the source.

        This is an abstract hook method that must be implemented by subclasses
        to define how data is read from a specific source.

        Returns:
            DataFrame containing the data
        """
        pass

    def _post_process(self, data: pl.DataFrame) -> pl.DataFrame:
        """Post-process the data after reading.

        This hook method can be overridden by subclasses to implement
        specific post-processing logic.

        Args:
            data: The data read from the source

        Returns:
            Processed DataFrame
        """
        return data


class FileReader(Reader):
    """Base class for file-based readers."""

    path: str

    def __str__(self) -> str:
        """Return string representation of the reader."""
        return f"{self.__class__.__name__}(path={self.path})"

    def _validate_source(self) -> None:
        """Validate that the file exists.

        Raises:
            FileNotFoundError: If the file does not exist
        """
        import os

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
