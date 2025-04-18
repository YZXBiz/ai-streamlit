"""JSON reader for data input."""

from pathlib import Path
from typing import Any, Union

import pandas as pd

from shared.io import Reader, load_json


class JSONReader(Reader):
    """Reader for JSON files."""

    def __init__(self, path: Union[str, Path], **kwargs: Any):
        """Initialize the JSON reader.
        
        Args:
            path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json
        """
        self.path = Path(path)
        self.kwargs = kwargs

    def read(self) -> Any:
        """Read data from JSON file.
        
        Returns:
            Data loaded from JSON
        """
        return load_json(self.path, **self.kwargs) 