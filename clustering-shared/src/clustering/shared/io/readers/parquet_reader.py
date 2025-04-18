"""Parquet reader implementation."""

import polars as pl

from clustering.shared.io.readers.base import FileReader


class ParquetReader(FileReader):
    """Reader for Parquet files."""

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from Parquet file.

        Returns:
            DataFrame containing the data
        """
        return pl.read_parquet(self.path)
