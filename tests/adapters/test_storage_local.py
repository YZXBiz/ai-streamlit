"""Tests for LocalFileStorage adapter."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from app.adapters.storage_local import LocalFileStorage
from app.domain.models.datafile import FileType


@pytest.fixture
def storage_dir():
    """Create a temporary directory for file storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def local_storage(storage_dir):
    """Create a LocalFileStorage instance for testing."""
    return LocalFileStorage(base_path=storage_dir)


def test_storage_initialization(local_storage, storage_dir):
    """Test that the LocalFileStorage initializes correctly."""
    assert local_storage.base_path == storage_dir
    # Check that the storage directory exists
    assert os.path.exists(storage_dir)


def test_save_file():
    """Test saving a file."""
    storage_path = "/tmp/test_storage"

    with (
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", mock_open()) as mock_file,
        patch("os.path.exists", return_value=False),
    ):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Mock file data
        file_data = MagicMock()
        file_data.read.return_value = b"test file content"
        file_data.filename = "test.csv"

        # Save the file
        result = storage.save_file(
            user_id=1,
            file_data=file_data,
            file_type=FileType.CSV,
        )

        # Check results
        assert "user_1" in result.file_path
        assert result.filename.endswith(".csv")
        assert result.original_filename == "test.csv"
        assert result.file_type == FileType.CSV
        assert result.file_size > 0

        # Check that directory was created
        mock_makedirs.assert_called_once()

        # Check that file was written
        mock_file.assert_called_once()


def test_save_existing_directory():
    """Test saving a file with existing directory."""
    storage_path = "/tmp/test_storage"

    with (
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", mock_open()) as mock_file,
        patch("os.path.exists", return_value=True),
    ):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Mock file data
        file_data = MagicMock()
        file_data.read.return_value = b"test file content"
        file_data.filename = "test.csv"

        # Save the file
        storage.save_file(
            user_id=1,
            file_data=file_data,
            file_type=FileType.CSV,
        )

        # Directory already exists, so makedirs should not be called
        mock_makedirs.assert_not_called()


def test_get_file():
    """Test retrieving a file."""
    storage_path = "/tmp/test_storage"
    file_path = "/tmp/test_storage/user_1/test_file.csv"

    with (
        patch("builtins.open", mock_open(read_data=b"test file content")) as mock_file,
        patch("os.path.exists", return_value=True),
    ):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Get the file
        file_data = storage.get_file(file_path)

        # Check results
        assert file_data == b"test file content"

        # Check that file was read
        mock_file.assert_called_once_with(file_path, "rb")


def test_get_nonexistent_file():
    """Test retrieving a nonexistent file."""
    storage_path = "/tmp/test_storage"
    file_path = "/tmp/test_storage/user_1/nonexistent.csv"

    with patch("os.path.exists", return_value=False):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Attempt to get a nonexistent file
        with pytest.raises(FileNotFoundError):
            storage.get_file(file_path)


def test_delete_file():
    """Test deleting a file."""
    storage_path = "/tmp/test_storage"
    file_path = "/tmp/test_storage/user_1/test_file.csv"

    with patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Delete the file
        result = storage.delete_file(file_path)

        # Check results
        assert result is True

        # Check that file was deleted
        mock_remove.assert_called_once_with(file_path)


def test_delete_nonexistent_file():
    """Test deleting a nonexistent file."""
    storage_path = "/tmp/test_storage"
    file_path = "/tmp/test_storage/user_1/nonexistent.csv"

    with patch("os.path.exists", return_value=False):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # Attempt to delete a nonexistent file
        result = storage.delete_file(file_path)

        # Should return False for nonexistent file
        assert result is False


def test_list_files():
    """Test listing files for a user."""
    storage_path = "/tmp/test_storage"
    user_id = 1
    user_dir = os.path.join(storage_path, f"user_{user_id}")

    # Mock files in the directory
    mock_files = ["file1.csv", "file2.xlsx", "file3.parquet"]
    mock_paths = [os.path.join(user_dir, f) for f in mock_files]

    with (
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=mock_files),
        patch("os.path.isfile", return_value=True),
        patch("os.path.join", side_effect=lambda *args: os.path.join(*args)),
    ):
        # Create the storage
        storage = LocalFileStorage(base_path=storage_path)

        # List files for the user
        files = storage.list_files(user_id)

        # Check results
        assert len(files) == 3
        assert all(f in mock_paths for f in files)
