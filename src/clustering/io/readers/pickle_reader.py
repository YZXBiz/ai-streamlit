"""Pickle reader implementation."""

import pickle

import polars as pl

from clustering.io.readers.base import FileReader


class PickleReader(FileReader):
    """Reader for Pickle files."""

    def read(self) -> pl.DataFrame:
        """Read data from Pickle file.

        Returns:
            DataFrame containing the data
        """
        with open(self.path, "rb") as file:
            data = pickle.load(file)

        # Convert to Polars DataFrame if needed
        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data) if hasattr(data, "to_pandas") else pl.DataFrame(data)

        if self.limit is not None:
            data = data.head(self.limit)

        return data
