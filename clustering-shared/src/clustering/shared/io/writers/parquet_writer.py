"""Parquet writer implementation."""

import polars as pl

from clustering.shared.io.writers.base import FileWriter


class ParquetWriter(FileWriter):
    """Writer for Parquet files."""

    compression: str | None = "snappy"
    use_pyarrow: bool = True

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to Parquet file.

        Args:
            data: Data to write
        """
        data.write_parquet(
            self.path,
            compression=self.compression,
            use_pyarrow=self.use_pyarrow,
        )
