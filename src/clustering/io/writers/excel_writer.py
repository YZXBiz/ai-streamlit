"""Excel writer implementation."""

import polars as pl

from clustering.io.writers.base import FileWriter


class ExcelWriter(FileWriter):
    """Writer for Excel files."""

    sheet_name: str = "Sheet1"
    engine: str = "openpyxl"

    def write(self, data: pl.DataFrame) -> None:
        """Write data to Excel file.

        Args:
            data: DataFrame to write
        """
        self._prepare_path()
        data.write_excel(self.path, sheet_name=self.sheet_name, engine=self.engine)
