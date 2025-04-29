"""Tests for the file service."""

import io
import os
import tempfile
from typing import BinaryIO
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from backend.app.domain.models.datafile import FileType
from backend.app.domain.models.file import DataFile
from backend.app.domain.models.user import User
from backend.app.ports.repository import DataFileRepository, FileRepository
from backend.app.ports.storage import FileStorage
from backend.app.services.file_service import FileService


@pytest.fixture
def mock_file_repo():
    """Create a mock file repository for testing."""
    repo = MagicMock(spec=DataFileRepository)

    # Setup the get method to return a DataFile
    repo.get.return_value = DataFile(
        id=1,
        user_id=1,
        filename="test_file.csv",
        original_filename="original.csv",
        file_path="/path/to/file.csv",
        file_size=1024,
        file_type=FileType.CSV,
        description="Test file",
    )

    # Setup the create method to return a DataFile with assigned ID
    repo.create.side_effect = lambda data_file: DataFile(
        id=1,
        user_id=data_file.user_id,
        filename=data_file.filename,
        original_filename=data_file.original_filename,
        file_path=data_file.file_path,
        file_size=data_file.file_size,
        file_type=data_file.file_type,
        description=data_file.description,
    )

    return repo


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for testing file storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


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

    @pytest.mark.service
    def test_init(self, mock_file_repo, temp_storage_path):
        """Test initialization of FileService."""
        service = FileService(file_repository=mock_file_repo, storage_path=temp_storage_path)

        assert service.file_repository == mock_file_repo
        assert service.storage_path == temp_storage_path
        assert service.file_storage is not None

    @pytest.mark.service
    @patch("app.services.file_service.uuid.uuid4")
    async def test_upload_file(self, mock_uuid, mock_file_repo, temp_storage_path):
        """Test uploading a file."""
        # Mock UUID for deterministic file names
        mock_uuid.return_value = "test-uuid-1234"

        service = FileService(file_repository=mock_file_repo, storage_path=temp_storage_path)

        # Create a temporary file to upload
        file_content = b"test,data\n1,2\n3,4"
        original_filename = "data.csv"

        # Create a mock UploadFile object
        mock_file = MagicMock()
        mock_file.filename = original_filename
        mock_file.content_type = "text/csv"
        mock_file.size = len(file_content)

        # Mock the read method to return file_content
        mock_file.read = MagicMock(return_value=file_content)

        # Call the upload_file method
        result = await service.upload_file(file=mock_file, user_id=1, description="Test upload")

        # Check that the result is a DataFile with expected properties
        assert isinstance(result, DataFile)
        assert result.user_id == 1
        assert result.original_filename == original_filename
        assert result.file_type == FileType.CSV
        assert result.description == "Test upload"

        # Check that the file was saved to disk
        expected_path = os.path.join(temp_storage_path, "1", "test-uuid-1234_data.csv")
        assert os.path.exists(expected_path)

        # Check file repository was called to create an entry
        mock_file_repo.create.assert_called_once()

        # Check the file contents
        with open(expected_path, "rb") as f:
            assert f.read() == file_content

    @pytest.mark.service
    async def test_get_file_by_id(self, mock_file_repo, temp_storage_path):
        """Test getting a file by ID."""
        service = FileService(file_repository=mock_file_repo, storage_path=temp_storage_path)

        file = await service.get_file_by_id(file_id=1)

        assert file is not None
        assert file.id == 1
        assert file.filename == "test_file.csv"

        # Verify repository was called
        mock_file_repo.get.assert_called_once_with(1)

    @pytest.mark.service
    async def test_get_files_by_user_id(self, mock_file_repo, temp_storage_path):
        """Test getting files by user ID."""
        # Setup mock to return a list of files
        mock_files = [
            DataFile(
                id=1,
                user_id=1,
                filename="file1.csv",
                original_filename="file1.csv",
                file_path="/path/to/file1.csv",
                file_size=1024,
                file_type=FileType.CSV,
                description="File 1",
            ),
            DataFile(
                id=2,
                user_id=1,
                filename="file2.csv",
                original_filename="file2.csv",
                file_path="/path/to/file2.csv",
                file_size=2048,
                file_type=FileType.CSV,
                description="File 2",
            ),
        ]
        mock_file_repo.get_by_user_id.return_value = mock_files

        service = FileService(file_repository=mock_file_repo, storage_path=temp_storage_path)

        files = await service.get_files_by_user_id(user_id=1)

        assert len(files) == 2
        assert files[0].id == 1
        assert files[1].id == 2

        # Verify repository was called
        mock_file_repo.get_by_user_id.assert_called_once_with(1)

    @pytest.mark.service
    @patch("app.services.file_service.os.remove")
    async def test_delete_file(self, mock_remove, mock_file_repo, temp_storage_path):
        """Test deleting a file."""
        service = FileService(file_repository=mock_file_repo, storage_path=temp_storage_path)

        result = await service.delete_file(file_id=1)

        assert result is True

        # Verify repository was called to get and delete
        mock_file_repo.get.assert_called_once_with(1)
        mock_file_repo.delete.assert_called_once_with(1)

        # Verify os.remove was called to delete the file
        mock_remove.assert_called_once_with("/path/to/file.csv")
