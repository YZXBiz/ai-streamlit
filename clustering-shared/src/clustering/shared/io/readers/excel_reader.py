"""Excel reader implementation."""

import polars as pl
import pandas as pd

from clustering.shared.io.readers.base import FileReader


class ExcelReader(FileReader):
    """Reader for Excel files."""

    sheet_name: str | int | None = None
    engine: str = "openpyxl"

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from Excel file.

        Returns:
            DataFrame containing the data
        """
        # Use pandas to read Excel and convert to polars
        # This is most reliable across different polars versions
        df_pandas = pd.read_excel(
            self.path,
            sheet_name=self.sheet_name,
            engine=self.engine,
        )
        return pl.from_pandas(df_pandas)

    def read(self) -> pl.DataFrame:
        """Read data from Excel file.

        Returns:
            DataFrame containing the data
        """
        data = self._read_from_source()

        if self.limit is not None:
            data = data.head(self.limit)

        return data
