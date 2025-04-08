"""Pickle writer implementation."""

import pickle

import polars as pl

from clustering.io.writers.base import FileWriter


class PickleWriter(FileWriter):
    """Writer for Pickle files.

    This writer supports writing both individual DataFrames and dictionaries
    of DataFrames to pickle files.
    """

    protocol: int = pickle.HIGHEST_PROTOCOL

    def _validate_data(self, data: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Validate the data before writing.

        This method overrides the base validation to support both DataFrames
        and dictionaries of DataFrames.

        Args:
            data: DataFrame or dictionary of DataFrames to validate

        Raises:
            ValueError: If the data is invalid
        """
        if isinstance(data, pl.DataFrame):
            # Handle DataFrame validation
            if data.height == 0:
                raise ValueError("Cannot write empty DataFrame")
        elif isinstance(data, dict):
            # Handle dictionary validation
            if not data:
                raise ValueError("Cannot write empty dictionary")

            # Validate that dictionary values are DataFrames
            for key, value in data.items():
                if not isinstance(value, pl.DataFrame):
                    raise ValueError(f"Dictionary value for key '{key}' is not a DataFrame")
                if value.height == 0:
                    raise ValueError(f"DataFrame for key '{key}' is empty")
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. Expected DataFrame or dictionary of DataFrames."
            )

    def _write_to_destination(self, data: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Write data to Pickle file.

        This implements the abstract method required by the Writer base class.
        Supports both DataFrames and dictionaries of DataFrames.

        Args:
            data: DataFrame or dictionary of DataFrames to write
        """
        with open(self.path, "wb") as file:
            pickle.dump(data, file, protocol=self.protocol)
