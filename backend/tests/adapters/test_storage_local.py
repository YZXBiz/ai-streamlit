"""Tests for the local file storage adapter."""

import os
import tempfile
import uuid
from pathlib import Path

import pytest
from backend.app.adapters.storage_local import LocalFileStorage


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for testing file storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestLocalFileStorage:
    """Test the LocalFileStorage adapter."""
    
    @pytest.mark.adapter
    def test_init(self, temp_storage_path):
        """Test initialization of LocalFileStorage."""
        storage = LocalFileStorage(temp_storage_path)
        assert storage.storage_path == Path(temp_storage_path)
        
    @pytest.mark.adapter
    def test_save_file(self, temp_storage_path):
        """Test saving a file."""
        storage = LocalFileStorage(temp_storage_path)
        content = b"Test file content"
        user_id = 1
        
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        result = storage.save_file(
            user_id=user_id,
            file_id=file_id,
            file_content=content,
            original_filename=original_filename
        )
        
        # Check that the result is a valid path
        assert isinstance(result, str)
        assert os.path.exists(result)
        
        # Check the content of the saved file
        with open(result, "rb") as f:
            assert f.read() == content
    
    @pytest.mark.adapter
    def test_get_file(self, temp_storage_path):
        """Test retrieving a file."""
        storage = LocalFileStorage(temp_storage_path)
        content = b"Test file content"
        user_id = 1
        
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        path = storage.save_file(
            user_id=user_id,
            file_id=file_id,
            file_content=content,
            original_filename=original_filename
        )
        
        # Retrieve the file content
        file_content = storage.get_file(path)
        assert file_content == content
    
    @pytest.mark.adapter
    def test_delete_file(self, temp_storage_path):
        """Test deleting a file."""
        storage = LocalFileStorage(temp_storage_path)
        content = b"Test file content"
        user_id = 1
        
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        path = storage.save_file(
            user_id=user_id,
            file_id=file_id,
            file_content=content,
            original_filename=original_filename
        )
        
        # Verify file exists
        assert os.path.exists(path)
        
        # Delete the file
        storage.delete_file(path)
        
        # Verify file no longer exists
        assert not os.path.exists(path)
    
    @pytest.mark.adapter
    def test_file_exists(self, temp_storage_path):
        """Test checking if a file exists."""
        storage = LocalFileStorage(temp_storage_path)
        content = b"Test file content"
        user_id = 1
        
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        path = storage.save_file(
            user_id=user_id,
            file_id=file_id,
            file_content=content,
            original_filename=original_filename
        )
        
        # Check that the file exists
        assert storage.file_exists(path)
        
        # Delete the file
        os.remove(path)
        
        # Check that the file no longer exists
        assert not storage.file_exists(path)
    
    @pytest.mark.adapter
    def test_get_file_size(self, temp_storage_path):
        """Test getting the file size."""
        storage = LocalFileStorage(temp_storage_path)
        content = b"Test file content"
        user_id = 1
        
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        path = storage.save_file(
            user_id=user_id,
            file_id=file_id,
            file_content=content,
            original_filename=original_filename
        )
        
        # Check the file size
        assert storage.get_file_size(path) == len(content)
    
    @pytest.mark.adapter
    def test_get_file_path(self, temp_storage_path):
        """Test generating a file path."""
        storage = LocalFileStorage(temp_storage_path)
        user_id = 1
        file_id = str(uuid.uuid4())
        original_filename = "test.txt"
        
        path = storage.get_file_path(user_id, file_id, original_filename)
        
        # Check that the path follows the expected format
        expected_path = Path(temp_storage_path) / str(user_id) / f"{file_id}_test.txt"
        assert Path(path) == expected_path 