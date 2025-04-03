"""Pickle writer implementation."""

import pickle

import polars as pl

from clustering.io.writers.base import FileWriter


class PickleWriter(FileWriter):
    """Writer for Pickle files."""

    protocol: int = pickle.HIGHEST_PROTOCOL

    def write(self, data: pl.DataFrame) -> None:
        """Write data to Pickle file.

        Args:
            data: DataFrame to write
        """
        self._prepare_path()
        with open(self.path, "wb") as file:
            pickle.dump(data, file, protocol=self.protocol)
