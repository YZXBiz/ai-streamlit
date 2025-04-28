"""Tests for FileStorage port."""

import pytest
from app.ports.storage import FileStorage


def test_filestorage_is_abstract():
    """Test that FileStorage is an abstract base class."""
    # Trying to instantiate FileStorage should raise TypeError
    with pytest.raises(TypeError):
        FileStorage()


def test_filestorage_abstract_methods():
    """Test that FileStorage has the expected abstract methods."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(FileStorage)
        if getattr(getattr(FileStorage, method_name), "__isabstractmethod__", False)
    ]

    # Check that all the expected methods are abstract
    expected_methods = ["save_file", "get_file_path", "delete_file", "file_exists", "get_file_size"]
    for method in expected_methods:
        assert method in abstract_methods


class ConcreteFileStorageForTesting(FileStorage):
    """A concrete implementation of FileStorage for testing."""

    async def save_file(self, file, user_id, custom_filename=None):
        return "test-filename"

    async def get_file_path(self, filename, user_id):
        return "/test/path/file.txt"

    async def delete_file(self, filename, user_id):
        return True

    async def file_exists(self, filename, user_id):
        return True

    async def get_file_size(self, filename, user_id):
        return 1024


def test_concrete_filestorage():
    """Test that a concrete implementation of FileStorage can be instantiated."""
    storage = ConcreteFileStorageForTesting()
    assert isinstance(storage, FileStorage)
