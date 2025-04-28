"""Abstract interface for file storage operations."""

from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import UploadFile


class FileStorage(ABC):
    """Abstract interface for file storage operations."""

    @abstractmethod
    async def save_file(
        self, file: UploadFile, user_id: int, custom_filename: str | None = None
    ) -> str:
        """
        Save an uploaded file to storage.

        Args:
            file: The uploaded file object
            user_id: ID of the user who uploaded the file
            custom_filename: Optional custom filename to use

        Returns:
            The path where the file was saved
        """
        pass

    @abstractmethod
    async def get_file_path(self, filename: str, user_id: int) -> Path:
        """
        Get the full path to a stored file.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            Path object for the file
        """
        pass

    @abstractmethod
    async def delete_file(self, filename: str, user_id: int) -> bool:
        """
        Delete a file from storage.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def file_exists(self, filename: str, user_id: int) -> bool:
        """
        Check if a file exists in storage.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            True if the file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_file_size(self, filename: str, user_id: int) -> int:
        """
        Get the size of a file in bytes.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            Size of the file in bytes
        """
        pass
