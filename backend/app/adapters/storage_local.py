"""
LocalFileStorage adapter for file storage.

This adapter implements the FileStorage interface to store files
on the local filesystem.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from ..ports.storage import FileStorage


class LocalFileStorage(FileStorage):
    """Local file system storage adapter."""

    def __init__(self, base_path: str):
        """
        Initialize the local file storage.

        Args:
            base_path: Base directory path for file storage
        """
        self.base_path = Path(base_path)
        self._ensure_base_path_exists()

    def _ensure_base_path_exists(self) -> None:
        """Ensure the base storage path exists."""
        os.makedirs(self.base_path, exist_ok=True)

    def _get_user_dir(self, user_id: int) -> Path:
        """Get the directory path for a specific user."""
        user_dir = self.base_path / f"user_{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    async def save_file(
        self, file: UploadFile, user_id: int, custom_filename: str | None = None
    ) -> str:
        """
        Save an uploaded file to the local storage.

        Args:
            file: The uploaded file object
            user_id: ID of the user who uploaded the file
            custom_filename: Optional custom filename to use

        Returns:
            The filename where the file was saved
        """
        user_dir = self._get_user_dir(user_id)

        # Generate a unique filename if none is provided
        if custom_filename is None:
            file_extension = self._get_file_extension(file.filename or "")
            filename = f"{uuid.uuid4().hex}{file_extension}"
        else:
            filename = custom_filename

        file_path = user_dir / filename

        # Save the file
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()

        return filename

    def _get_file_extension(self, filename: str) -> str:
        """Extract the file extension from the filename."""
        if "." in filename:
            return f".{filename.split('.')[-1]}"
        return ""

    async def get_file_path(self, filename: str, user_id: int) -> Path:
        """
        Get the full path to a stored file.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            Path object for the file
        """
        user_dir = self._get_user_dir(user_id)
        file_path = user_dir / filename

        # Check if the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} does not exist for user {user_id}")

        return file_path

    async def delete_file(self, filename: str, user_id: int) -> bool:
        """
        Delete a file from storage.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = await self.get_file_path(filename, user_id)
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    async def file_exists(self, filename: str, user_id: int) -> bool:
        """
        Check if a file exists in storage.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            True if the file exists, False otherwise
        """
        user_dir = self._get_user_dir(user_id)
        file_path = user_dir / filename
        return file_path.exists()

    async def get_file_size(self, filename: str, user_id: int) -> int:
        """
        Get the size of a file in bytes.

        Args:
            filename: Name of the file
            user_id: ID of the user who owns the file

        Returns:
            Size of the file in bytes
        """
        try:
            file_path = await self.get_file_path(filename, user_id)
            return file_path.stat().st_size
        except FileNotFoundError:
            return 0
