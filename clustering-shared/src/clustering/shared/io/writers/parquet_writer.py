"""Parquet writer implementation."""

import polars as pl

from clustering.shared.io.writers.base import FileWriter


class ParquetWriter(FileWriter):
    """Writer for Parquet files."""

    compression: str = "snappy"
    use_pyarrow: bool = True

    def write(self, data: pl.DataFrame) -> None:
        """Write data to Parquet file.

        Args:
            data: DataFrame to write
        """
        self._prepare_path()
        data.write_parquet(self.path, compression=self.compression, use_pyarrow=self.use_pyarrow)
