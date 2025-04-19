"""Excel writer implementation."""

import polars as pl

from clustering.shared.io.writers.base import FileWriter


class ExcelWriter(FileWriter):
    """Writer for Excel files."""

    sheet_name: str = "Sheet1"
    engine: str = "openpyxl"
    index: bool = False

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to Excel file.

        Args:
            data: Data to write
        """
        # Convert to pandas first as polars write_excel may not be fully implemented
        pandas_df = data.to_pandas()
        pandas_df.to_excel(
            self.path,
            sheet_name=self.sheet_name,
            engine=self.engine,
            index=self.index,
        )

    def write(self, data: pl.DataFrame) -> None:
        """Write data to Excel file.

        Args:
            data: DataFrame to write
        """
        self._prepare_for_writing()
        self._write_to_destination(data)
