"""JSON writer for data output."""

import json

import polars as pl

from clustering.shared.io.writers.base import FileWriter


class JSONWriter(FileWriter):
    """Writer for JSON files."""

    orient: str = "records"
    lines: bool = True
    pretty: bool = False

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write DataFrame to a JSON file.

        Args:
            data: DataFrame to write
        """
        # Polars' JSON writer is limited in options
        # We need to handle options manually
        if self.lines:
            # Write as JSON lines (newline-delimited JSON)
            records = data.to_dicts()
            with open(self.path, "w") as f:
                for record in records:
                    if self.pretty:
                        json_str = json.dumps(record, indent=2)
                    else:
                        json_str = json.dumps(record)
                    f.write(json_str + "\n")
        else:
            # Write as a single JSON array
            if self.orient == "records":
                records = data.to_dicts()
                with open(self.path, "w") as f:
                    if self.pretty:
                        json.dump(records, f, indent=2)
                    else:
                        json.dump(records, f)
            else:
                # Other orient options would be implemented here if needed
                raise ValueError(f"Orient option '{self.orient}' not supported")
