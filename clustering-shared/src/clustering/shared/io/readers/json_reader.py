"""JSON reader for data input."""

import json

import polars as pl

from clustering.shared.io.readers.base import FileReader


class JSONReader(FileReader):
    """Reader for JSON files."""

    lines: bool = True

    def _read_from_source(self) -> pl.DataFrame:
        """Read data from JSON file.

        Returns:
            DataFrame with data from JSON file
        """
        # Polars doesn't support a lines parameter for read_json directly
        # For lines=True, we need to handle JSON lines (newline-delimited JSON)
        with open(self.path, "r") as f:
            content = f.read()

        if self.lines:
            # For JSON lines, each line is a JSON object
            if not content.strip():
                # Handle empty file
                return pl.DataFrame()

            # Parse each line as separate JSON object and create DataFrame
            json_objects = [
                json.loads(line) for line in content.strip().split("\n") if line.strip()
            ]
            return pl.DataFrame(json_objects)
        else:
            # For regular JSON, assume array of objects
            if not content.strip():
                # Handle empty file
                return pl.DataFrame()

            # Parse JSON and create DataFrame
            return pl.DataFrame(json.loads(content))
