"""CSV reader implementation."""

import polars as pl

from clustering.io.readers.base import FileReader


class CSVReader(FileReader):
    """Reader for CSV files."""

    delimiter: str = ","
    has_header: bool = True

    def read(self) -> pl.DataFrame:
        """Read data from CSV file.

        Returns:
            DataFrame containing the data
        """
        data = pl.read_csv(
            self.path,
            separator=self.delimiter,
            has_header=self.has_header,
        )

        if self.limit is not None:
            data = data.head(self.limit)

        return data
