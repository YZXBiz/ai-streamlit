"""Excel reader implementation."""

import polars as pl

from clustering.shared.io.readers.base import FileReader


class ExcelReader(FileReader):
    """Reader for Excel files."""

    sheet_name: str | int | None = 0
    engine: str = "openpyxl"

    def read(self) -> pl.DataFrame:
        """Read data from Excel file.

        Returns:
            DataFrame containing the data
        """
        data = pl.read_excel(self.path, sheet_name=self.sheet_name, engine=self.engine)

        if self.limit is not None:
            data = data.head(self.limit)

        return data
