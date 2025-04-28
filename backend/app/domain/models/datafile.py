"""DataFile domain model for storing metadata about uploaded data files."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class FileType(Enum):
    """Enum for file types supported by the application."""

    CSV = auto()
    EXCEL = auto()
    PARQUET = auto()
    JSON = auto()


@dataclass
class DataFile:
    """DataFile domain model representing an uploaded data file."""

    id: int | None = None
    user_id: int | None = None
    filename: str = ""
    original_filename: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: FileType = FileType.CSV
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_file_extension(self) -> str:
        """Return the file extension based on file type."""
        if self.file_type == FileType.CSV:
            return ".csv"
        elif self.file_type == FileType.EXCEL:
            return ".xlsx"
        elif self.file_type == FileType.PARQUET:
            return ".parquet"
        elif self.file_type == FileType.JSON:
            return ".json"
        return ""
