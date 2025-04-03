"""CSV reader implementation."""

import polars as pl

from clustering.io.readers.base import FileReader


class CSVReader(FileReader):
    """Reader for CSV files."""

    delimiter: str = ","
    has_header: bool = True

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from CSV file.

        Returns:
            DataFrame containing the data
        """
        return pl.read_csv(
            self.path,
            separator=self.delimiter,
            has_header=self.has_header,
        )
