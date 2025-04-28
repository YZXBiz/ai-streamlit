"""Tests for the file service."""

import io
import os
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from backend.app.domain.models.file import DataFile
from backend.app.domain.models.user import User
from backend.app.ports.repository import FileRepository
from backend.app.ports.storage import FileStorage
from backend.app.services.file_service import FileService


@pytest.mark.asyncio
class TestFileService:
    """Tests for the FileService class."""

    def setup_method(self):
        """Set up test environment."""
        self.file_repo = AsyncMock(spec=FileRepository)
        self.file_storage = AsyncMock(spec=FileStorage)
        self.service = FileService(
            file_repository=self.file_repo,
            file_storage=self.file_storage,
        )

        # Create a mock user for testing
        self.user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_upload_file_csv(self):
        """Test uploading a CSV file."""
        # Create mock file data
        file_content = io.BytesIO(b"col1,col2\n1,2\n3,4")

        # Create a mock UploadFile
        mock_file = MagicMock()
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.size = 1024
        mock_file.file = file_content

        # Configure mock storage
        storage_path = f"/uploads/user_{self.user.id}/test_data.csv"
        self.file_storage.save_file.return_value = storage_path

        # Configure mock repository
        mock_data_file = DataFile(
            id=1,
            user_id=self.user.id,
            filename="test_data.csv",
            file_path=storage_path,
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.create_file.return_value = mock_data_file

        # Call the service method
        result = await self.service.upload_file(self.user, mock_file)

        # Verify file was saved
        self.file_storage.save_file.assert_called_once()
        # Check user_id and filename in the arguments
        args, kwargs = self.file_storage.save_file.call_args
        assert args[0] == self.user.id
        assert "test_data.csv" in args[1]
        assert isinstance(args[2], io.BytesIO)

        # Verify file record was created
        self.file_repo.create_file.assert_called_once()
        create_args, _ = self.file_repo.create_file.call_args
        assert create_args[0].user_id == self.user.id
        assert create_args[0].filename == "test_data.csv"
        assert create_args[0].file_path == storage_path

        # Verify result
        assert result.id == 1
        assert result.user_id == self.user.id
        assert result.filename == "test_data.csv"
        assert result.file_path == storage_path

    @pytest.mark.asyncio
    async def test_upload_file_excel(self):
        """Test uploading an Excel file."""
        # Create mock file data
        file_content = io.BytesIO(b"mock excel content")

        # Create a mock UploadFile
        mock_file = MagicMock()
        mock_file.filename = "test_data.xlsx"
        mock_file.content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        mock_file.size = 2048
        mock_file.file = file_content

        # Configure mock storage
        storage_path = f"/uploads/user_{self.user.id}/test_data.xlsx"
        self.file_storage.save_file.return_value = storage_path

        # Configure mock repository
        mock_data_file = DataFile(
            id=2,
            user_id=self.user.id,
            filename="test_data.xlsx",
            file_path=storage_path,
            file_size=2048,
            file_type="xlsx",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.create_file.return_value = mock_data_file

        # Call the service method
        result = await self.service.upload_file(self.user, mock_file)

        # Verify file was saved
        self.file_storage.save_file.assert_called_once()

        # Verify file record was created
        self.file_repo.create_file.assert_called_once()

        # Verify result
        assert result.id == 2
        assert result.user_id == self.user.id
        assert result.filename == "test_data.xlsx"
        assert result.file_type == "xlsx"

    @pytest.mark.asyncio
    async def test_upload_file_invalid_format(self):
        """Test uploading a file with an invalid format."""
        # Create mock file data
        file_content = io.BytesIO(b"invalid file content")

        # Create a mock UploadFile
        mock_file = MagicMock()
        mock_file.filename = "test_data.txt"
        mock_file.content_type = "text/plain"
        mock_file.file = file_content

        # Call the service method and expect an error
        with pytest.raises(ValueError) as exc_info:
            await self.service.upload_file(self.user, mock_file)

        # Verify error message
        assert "Unsupported file format" in str(exc_info.value)

        # Verify storage and repository were not called
        self.file_storage.save_file.assert_not_called()
        self.file_repo.create_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_user_files(self):
        """Test retrieving all files for a user."""
        # Configure mock repository
        mock_files = [
            DataFile(
                id=1,
                user_id=self.user.id,
                filename="file1.csv",
                file_path=f"/uploads/user_{self.user.id}/file1.csv",
                file_size=1024,
                file_type="csv",
                created_at="2023-01-01T12:00:00",
                updated_at="2023-01-01T12:00:00",
            ),
            DataFile(
                id=2,
                user_id=self.user.id,
                filename="file2.xlsx",
                file_path=f"/uploads/user_{self.user.id}/file2.xlsx",
                file_size=2048,
                file_type="xlsx",
                created_at="2023-01-02T12:00:00",
                updated_at="2023-01-02T12:00:00",
            ),
        ]
        self.file_repo.get_files_by_user.return_value = mock_files

        # Call the service method
        result = await self.service.get_user_files(self.user)

        # Verify repository was called
        self.file_repo.get_files_by_user.assert_called_once_with(user_id=self.user.id)

        # Verify result
        assert len(result) == 2
        assert result[0].id == 1
        assert result[0].filename == "file1.csv"
        assert result[1].id == 2
        assert result[1].filename == "file2.xlsx"

    @pytest.mark.asyncio
    async def test_get_file_success(self):
        """Test retrieving a specific file."""
        # Configure mock repository
        mock_file = DataFile(
            id=1,
            user_id=self.user.id,
            filename="test_data.csv",
            file_path=f"/uploads/user_{self.user.id}/test_data.csv",
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.get_file_by_id.return_value = mock_file

        # Call the service method
        result = await self.service.get_file(file_id=1, user=self.user)

        # Verify repository was called
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=1)

        # Verify result
        assert result.id == 1
        assert result.user_id == self.user.id
        assert result.filename == "test_data.csv"

    @pytest.mark.asyncio
    async def test_get_file_unauthorized(self):
        """Test retrieving a file that belongs to another user."""
        # Configure mock repository to return a file owned by a different user
        other_user_id = 999
        mock_file = DataFile(
            id=1,
            user_id=other_user_id,  # Different user ID
            filename="test_data.csv",
            file_path=f"/uploads/user_{other_user_id}/test_data.csv",
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.get_file_by_id.return_value = mock_file

        # Call the service method and expect an error
        with pytest.raises(Exception) as exc_info:
            await self.service.get_file(file_id=1, user=self.user)

        # Verify error message
        assert "Unauthorized access" in str(exc_info.value)

        # Verify repository was called
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=1)

    @pytest.mark.asyncio
    async def test_get_file_not_found(self):
        """Test retrieving a non-existent file."""
        # Configure mock repository to return None (file not found)
        self.file_repo.get_file_by_id.return_value = None

        # Call the service method and expect an error
        with pytest.raises(Exception) as exc_info:
            await self.service.get_file(file_id=999, user=self.user)

        # Verify error message
        assert "File not found" in str(exc_info.value)

        # Verify repository was called
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=999)

    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        """Test deleting a file."""
        # Configure mock repository
        mock_file = DataFile(
            id=1,
            user_id=self.user.id,
            filename="test_data.csv",
            file_path=f"/uploads/user_{self.user.id}/test_data.csv",
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.get_file_by_id.return_value = mock_file
        self.file_repo.delete_file.return_value = None

        # Configure mock storage
        self.file_storage.delete_file.return_value = None

        # Call the service method
        await self.service.delete_file(file_id=1, user=self.user)

        # Verify repository calls
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=1)
        self.file_repo.delete_file.assert_called_once_with(file_id=1)

        # Verify storage call
        self.file_storage.delete_file.assert_called_once_with(file_path=mock_file.file_path)

    @pytest.mark.asyncio
    async def test_delete_file_unauthorized(self):
        """Test deleting a file that belongs to another user."""
        # Configure mock repository to return a file owned by a different user
        other_user_id = 999
        mock_file = DataFile(
            id=1,
            user_id=other_user_id,  # Different user ID
            filename="test_data.csv",
            file_path=f"/uploads/user_{other_user_id}/test_data.csv",
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.get_file_by_id.return_value = mock_file

        # Call the service method and expect an error
        with pytest.raises(Exception) as exc_info:
            await self.service.delete_file(file_id=1, user=self.user)

        # Verify error message
        assert "Unauthorized access" in str(exc_info.value)

        # Verify repository was called for get but not for delete
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=1)
        self.file_repo.delete_file.assert_not_called()
        self.file_storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_file_content(self):
        """Test retrieving file content."""
        # Configure mock repository
        file_path = f"/uploads/user_{self.user.id}/test_data.csv"
        mock_file = DataFile(
            id=1,
            user_id=self.user.id,
            filename="test_data.csv",
            file_path=file_path,
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.file_repo.get_file_by_id.return_value = mock_file

        # Mock file content
        file_content = b"col1,col2\n1,2\n3,4"
        self.file_storage.get_file_content.return_value = file_content

        # Call the service method
        content, filename, content_type = await self.service.get_file_content(
            file_id=1, user=self.user
        )

        # Verify repository call
        self.file_repo.get_file_by_id.assert_called_once_with(file_id=1)

        # Verify storage call
        self.file_storage.get_file_content.assert_called_once_with(file_path=file_path)

        # Verify result
        assert content == file_content
        assert filename == "test_data.csv"
        assert content_type == "text/csv"
