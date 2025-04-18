"""Pickle reader implementation."""

import pickle

import polars as pl

from clustering.shared.io.readers.base import FileReader


class PickleReader(FileReader):
    """Reader for Pickle files."""

    def _read_from_source(self) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Read data from Pickle file.

        Returns:
            DataFrame or dictionary of DataFrames containing the data
        """
        with open(self.path, "rb") as file:
            data = pickle.load(file)

        # If data is already a dictionary of DataFrames, process each DataFrame
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Convert to Polars DataFrame if needed
                if not isinstance(value, pl.DataFrame):
                    value = (
                        pl.from_pandas(value)
                        if hasattr(value, "to_pandas")
                        else pl.DataFrame(value)
                    )
                result[key] = value
            return result

        # Otherwise, handle as a single DataFrame
        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data) if hasattr(data, "to_pandas") else pl.DataFrame(data)

        return data

    def read(self) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Override the read method to handle dictionary of DataFrames.

        The base read method is designed to return a single DataFrame,
        but we need to handle dictionaries of DataFrames as well.

        Returns:
            DataFrame or dictionary of DataFrames containing the data
        """
        # Step 1: Validate the source
        self._validate_source()

        # Step 2: Read data from source (implemented by subclasses)
        data = self._read_from_source()

        # Step 3: If it's a dictionary, apply limit to each DataFrame
        if isinstance(data, dict) and self.limit is not None:
            return {key: value.head(self.limit) for key, value in data.items()}
        # Otherwise, let the base implementation handle it
        elif not isinstance(data, dict) and self.limit is not None:
            data = data.head(self.limit)

        # Step 4: Post-process the data (only if it's a single DataFrame)
        if not isinstance(data, dict):
            data = self._post_process(data)

        return data
