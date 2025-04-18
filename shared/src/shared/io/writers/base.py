"""Base classes for data writers."""

from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import pydantic as pdt

from shared.common.filesystem import ensure_directory


class Writer(pdt.BaseModel, ABC):
    """Base class for data writers."""

    def write(self, data: pl.DataFrame) -> None:
        """Template method defining the writing algorithm.

        The template method pattern defines the skeleton of the writing algorithm,
        deferring some steps to subclasses. This ensures a consistent writing
        process while allowing specific implementations to vary.

        Args:
            data: DataFrame to write
        """
        # Step 1: Validate the data
        self._validate_data(data)

        # Step 2: Prepare for writing
        self._prepare_for_writing()

        # Step 3: Write data to destination
        self._write_to_destination(data)

        # Step 4: Post-process
        self._post_process()

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate the data before writing.

        This hook method can be overridden by subclasses to implement
        specific validation logic.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If the data is invalid
        """
        if data.height == 0:
            raise ValueError("Cannot write empty DataFrame")

    def _prepare_for_writing(self) -> None:
        """Prepare for writing.

        This hook method can be overridden by subclasses to implement
        specific preparation logic.
        """
        pass

    @abstractmethod
    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to the destination.

        This is an abstract hook method that must be implemented by subclasses
        to define how data is written to a specific destination.

        Args:
            data: DataFrame to write
        """
        pass

    def _post_process(self) -> None:
        """Post-process after writing.

        This hook method can be overridden by subclasses to implement
        specific post-processing logic.
        """
        pass


class FileWriter(Writer):
    """Base class for file-based writers."""

    path: str
    create_parent_dirs: bool = True

    def __str__(self) -> str:
        """Return string representation of the writer."""
        return f"{self.__class__.__name__}(path={self.path})"

    def _prepare_for_writing(self) -> None:
        """Prepare the path by creating parent directories if needed."""
        if self.create_parent_dirs:
            ensure_directory(Path(self.path).parent)
