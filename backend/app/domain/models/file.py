"""File domain model for storing file data."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FileType(Enum):
    """Enum for supported file types."""

    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSON = "json"
    SQL = "sql"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, extension: str) -> "FileType":
        """Determine file type from file extension."""
        extension = extension.lower().strip(".")
        mapping = {
            "csv": cls.CSV,
            "xlsx": cls.EXCEL,
            "xls": cls.EXCEL,
            "parquet": cls.PARQUET,
            "json": cls.JSON,
            "sql": cls.SQL,
        }
        return mapping.get(extension, cls.UNKNOWN)

    @property
    def is_tabular(self) -> bool:
        """Check if this file type is tabular data."""
        return self in [self.CSV, self.EXCEL, self.PARQUET]


@dataclass
class DataFile:
    """DataFile domain model representing a file uploaded by a user."""

    id: int | None = None
    user_id: int | None = None
    filename: str = ""  # System filename (stored on disk)
    original_filename: str = ""  # Original user filename
    file_path: str = ""  # Full path to the file on disk
    file_type: FileType = FileType.UNKNOWN
    file_size: int = 0  # Size in bytes
    row_count: int | None = None
    column_count: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_tabular(self) -> bool:
        """Check if this file contains tabular data."""
        return self.file_type.is_tabular

    @classmethod
    def from_upload(
        cls, user_id: int, original_filename: str, filename: str, file_path: str, file_size: int
    ) -> "DataFile":
        """Create a DataFile from upload information."""
        extension = original_filename.split(".")[-1] if "." in original_filename else ""
        file_type = FileType.from_extension(extension)

        return cls(
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
        )
