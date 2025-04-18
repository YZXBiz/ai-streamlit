"""CSV reader implementation."""

import polars as pl

from shared.io.readers.base import FileReader


class CSVReader(FileReader):
    """Reader for CSV files."""

    delimiter: str = ","
    has_header: bool = True
    quote_char: str = '"'
    ignore_errors: bool = True
    infer_schema_length: int = 10000
    try_parse_dates: bool = True
    null_values: list[str] = [""]

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from CSV file.

        Returns:
            DataFrame containing the data
        """
        try:
            return pl.read_csv(
                self.path,
                separator=self.delimiter,
                has_header=self.has_header,
                quote_char=self.quote_char,
                ignore_errors=self.ignore_errors,
                infer_schema_length=self.infer_schema_length,
                try_parse_dates=self.try_parse_dates,
                null_values=self.null_values,
            )
        except Exception:
            # Fallback to pandas and convert to polars if polars reader fails
            import pandas as pd

            df_pandas = pd.read_csv(
                self.path,
                sep=self.delimiter,
                header=0 if self.has_header else None,
                quotechar=self.quote_char,
                on_bad_lines="skip" if self.ignore_errors else "error",
            )
            return pl.from_pandas(df_pandas)
