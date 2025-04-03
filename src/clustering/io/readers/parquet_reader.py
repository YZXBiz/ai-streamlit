"""Parquet reader implementation."""

import polars as pl

from clustering.io.readers.base import FileReader


class ParquetReader(FileReader):
    """Reader for Parquet files."""

    def read(self) -> pl.DataFrame:
        """Read data from Parquet file.

        Returns:
            DataFrame containing the data
        """
        data = pl.read_parquet(self.path)

        if self.limit is not None:
            data = data.head(self.limit)

        return data
