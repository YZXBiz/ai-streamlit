"""CSV reader implementation."""

import polars as pl

from clustering.shared.io.readers.base import FileReader


class CSVReader(FileReader):
    """Reader for CSV files."""

    delimiter: str = ","
    has_header: bool = True
    quote_char: str = '"'
    ignore_errors: bool = True
    infer_schema_length: int = 10000
    try_parse_dates: bool = True
    null_values: list[str] = ["", "NA", "N/A", "None", "null"]
    columns: list[str] | None = None
    comment_char: str | None = None
    skip_rows: int = 0
    dtypes: dict[str, str] | None = None
    encoding: str = "utf8"

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from CSV file.

        Returns:
            DataFrame containing the data
        """
        try:
            # First attempt: Using polars with all parameters
            try:
                df = pl.read_csv(
                    self.path,
                    separator=self.delimiter,
                    has_header=self.has_header,
                    quote_char=self.quote_char,
                    ignore_errors=self.ignore_errors,
                    infer_schema_length=self.infer_schema_length,
                    try_parse_dates=self.try_parse_dates,
                    null_values=self.null_values,
                    skip_rows=self.skip_rows,
                    encoding=self.encoding,
                )

                # Filter columns if specified
                if self.columns is not None:
                    df = df.select([col for col in self.columns if col in df.columns])

                return df
            except Exception as e:
                # Log the error and try alternative approach
                print(f"Polars CSV reading failed: {e}. Trying fallback method...")

            # Second attempt: Try pandas with more lenient options
            import pandas as pd

            df_pandas = pd.read_csv(
                self.path,
                sep=self.delimiter,
                header=0 if self.has_header else None,
                quotechar=self.quote_char,
                on_bad_lines="skip" if self.ignore_errors else "error",
                skiprows=self.skip_rows,
                usecols=self.columns,
                dtype=self.dtypes,
                encoding=self.encoding,
                na_values=self.null_values,
                comment=self.comment_char,
                low_memory=False,  # More permissive reading
            )

            # Force column names to strings
            df_pandas.columns = df_pandas.columns.astype(str)

            return pl.from_pandas(df_pandas)
        except Exception as e:
            # If all reading methods fail, provide clear error message
            raise ValueError(f"Failed to read CSV file {self.path}: {str(e)}") from e
