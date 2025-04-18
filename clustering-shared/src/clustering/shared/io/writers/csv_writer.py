"""CSV writer implementation."""

import polars as pl

from shared.io.writers.base import FileWriter


class CSVWriter(FileWriter):
    """Writer for CSV files."""

    delimiter: str = ","
    include_header: bool = True
    include_bom: bool = False

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to CSV file.

        Args:
            data: DataFrame to write
        """
        data.write_csv(
            file=self.path,
            separator=self.delimiter,
            include_header=self.include_header,
            include_bom=self.include_bom,
        )
