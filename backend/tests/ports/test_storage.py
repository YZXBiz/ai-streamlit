"""Tests for storage module."""

import inspect
from abc import ABC
from pathlib import Path

import pytest
from fastapi import UploadFile


def test_storage_constants():
    """Test storage module constants."""
    from backend.app.ports.storage import FileStorage

    assert hasattr(FileStorage, "__abstractmethods__")


class TestFileStorage:
    """Tests for the FileStorage interface."""

    @pytest.mark.port
    def test_file_storage_is_abc(self):
        """Test that FileStorage is an ABC."""
        from backend.app.ports.storage import FileStorage

        assert issubclass(FileStorage, ABC)

    @pytest.mark.port
    def test_file_storage_methods(self):
        """Test that FileStorage has the expected methods."""
        from backend.app.ports.storage import FileStorage

        # Check required methods
        assert hasattr(FileStorage, "save_file")
        assert hasattr(FileStorage, "get_file_path")
        assert hasattr(FileStorage, "delete_file")
        assert hasattr(FileStorage, "file_exists")
        assert hasattr(FileStorage, "get_file_size")

    @pytest.mark.port
    def test_save_file_signature(self):
        """Test that save_file has the expected signature."""
        from backend.app.ports.storage import FileStorage

        sig = inspect.signature(FileStorage.save_file)
        params = sig.parameters

        assert "self" in params
        assert "file" in params
        assert "user_id" in params
        assert "custom_filename" in params

        # Check parameter types from type annotations
        assert params["file"].annotation == UploadFile
        assert params["user_id"].annotation == int
        assert str(params["custom_filename"].annotation) == "str | None"
        assert params["custom_filename"].default is None

        # Check return type
        assert sig.return_annotation == str

    @pytest.mark.port
    def test_get_file_path_signature(self):
        """Test that get_file_path has the expected signature."""
        from backend.app.ports.storage import FileStorage

        sig = inspect.signature(FileStorage.get_file_path)
        params = sig.parameters

        assert "self" in params
        assert "filename" in params
        assert "user_id" in params

        # Check parameter types
        assert params["filename"].annotation == str
        assert params["user_id"].annotation == int

        # Check return type
        assert sig.return_annotation == Path

    @pytest.mark.port
    def test_delete_file_signature(self):
        """Test that delete_file has the expected signature."""
        from backend.app.ports.storage import FileStorage

        sig = inspect.signature(FileStorage.delete_file)
        params = sig.parameters

        assert "self" in params
        assert "filename" in params
        assert "user_id" in params

        # Check parameter types
        assert params["filename"].annotation == str
        assert params["user_id"].annotation == int

        # Check return type
        assert sig.return_annotation == bool

    @pytest.mark.port
    def test_file_exists_signature(self):
        """Test that file_exists has the expected signature."""
        from backend.app.ports.storage import FileStorage

        sig = inspect.signature(FileStorage.file_exists)
        params = sig.parameters

        assert "self" in params
        assert "filename" in params
        assert "user_id" in params

        # Check parameter types
        assert params["filename"].annotation == str
        assert params["user_id"].annotation == int

        # Check return type
        assert sig.return_annotation == bool

    @pytest.mark.port
    def test_get_file_size_signature(self):
        """Test that get_file_size has the expected signature."""
        from backend.app.ports.storage import FileStorage

        sig = inspect.signature(FileStorage.get_file_size)
        params = sig.parameters

        assert "self" in params
        assert "filename" in params
        assert "user_id" in params

        # Check parameter types
        assert params["filename"].annotation == str
        assert params["user_id"].annotation == int

        # Check return type
        assert sig.return_annotation == int
