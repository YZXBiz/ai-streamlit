"""Tests for file upload API endpoints."""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from backend.app.api.routers.files import router
from backend.app.domain.models.datafile import DataFile
from backend.app.domain.models.user import User
from backend.app.main import app


@pytest.fixture
def test_client():
    """Create test client for the API."""
    return TestClient(app)


class TestFileRoutes:
    """Test suite for file routes."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock user for testing
        self.mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
        )

        # Create a mock file service
        self.mock_file_service = AsyncMock()

        # Configure app to use test dependencies
        app.dependency_overrides = {}

        # Override dependencies
        async def get_current_user():
            return self.mock_user

        async def get_file_service():
            return self.mock_file_service

        app.dependency_overrides["get_current_user"] = get_current_user
        app.dependency_overrides["get_file_service"] = get_file_service

    def teardown_method(self):
        """Clean up after tests."""
        # Reset dependency overrides
        app.dependency_overrides = {}

    def test_upload_file_success(self, test_client):
        """Test successful file upload."""
        # Create a mock file
        file_content = b"col1,col2\n1,2\n3,4"
        file = io.BytesIO(file_content)

        # Configure mock service response
        mock_file = DataFile(
            id=1,
            user_id=self.mock_user.id,
            filename="test.csv",
            file_path="/uploads/user_1/test.csv",
            file_size=len(file_content),
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.mock_file_service.upload_file.return_value = mock_file

        # Make the request
        response = test_client.post(
            "/api/files/",
            files={"file": ("test.csv", file, "text/csv")},
        )

        # Check response
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == 1
        assert data["filename"] == "test.csv"
        assert data["file_type"] == "csv"
        assert data["file_size"] == len(file_content)

        # Verify service was called
        self.mock_file_service.upload_file.assert_called_once()

    def test_upload_file_invalid_format(self, test_client):
        """Test uploading a file with invalid format."""
        # Create a mock file with invalid format
        file_content = b"This is a plain text file"
        file = io.BytesIO(file_content)

        # Configure mock service to raise exception
        self.mock_file_service.upload_file.side_effect = ValueError("Unsupported file format")

        # Make the request
        response = test_client.post(
            "/api/files/",
            files={"file": ("test.txt", file, "text/plain")},
        )

        # Check response
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file format" in data["detail"]

    def test_get_user_files(self, test_client):
        """Test retrieving files for a user."""
        # Configure mock service response
        mock_files = [
            DataFile(
                id=1,
                user_id=self.mock_user.id,
                filename="file1.csv",
                file_path="/uploads/user_1/file1.csv",
                file_size=1024,
                file_type="csv",
                created_at="2023-01-01T12:00:00",
                updated_at="2023-01-01T12:00:00",
            ),
            DataFile(
                id=2,
                user_id=self.mock_user.id,
                filename="file2.xlsx",
                file_path="/uploads/user_1/file2.xlsx",
                file_size=2048,
                file_type="xlsx",
                created_at="2023-01-02T12:00:00",
                updated_at="2023-01-02T12:00:00",
            ),
        ]
        self.mock_file_service.get_user_files.return_value = mock_files

        # Make the request
        response = test_client.get("/api/files/")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[0]["filename"] == "file1.csv"
        assert data[1]["id"] == 2
        assert data[1]["filename"] == "file2.xlsx"

        # Verify service was called
        self.mock_file_service.get_user_files.assert_called_once_with(user=self.mock_user)

    def test_get_file(self, test_client):
        """Test retrieving a specific file."""
        # Configure mock service response
        mock_file = DataFile(
            id=1,
            user_id=self.mock_user.id,
            filename="test.csv",
            file_path="/uploads/user_1/test.csv",
            file_size=1024,
            file_type="csv",
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )
        self.mock_file_service.get_file.return_value = mock_file

        # Make the request
        response = test_client.get("/api/files/1")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["filename"] == "test.csv"
        assert data["file_type"] == "csv"

        # Verify service was called
        self.mock_file_service.get_file.assert_called_once_with(file_id=1, user=self.mock_user)

    def test_get_file_not_found(self, test_client):
        """Test retrieving a non-existent file."""
        # Configure mock service to raise exception
        self.mock_file_service.get_file.side_effect = Exception("File not found")

        # Make the request
        response = test_client.get("/api/files/999")

        # Check response
        assert response.status_code == 404
        data = response.json()
        assert "File not found" in data["detail"]

    def test_delete_file(self, test_client):
        """Test deleting a file."""
        # Configure mock service
        self.mock_file_service.delete_file.return_value = None

        # Make the request
        response = test_client.delete("/api/files/1")

        # Check response
        assert response.status_code == 204

        # Verify service was called
        self.mock_file_service.delete_file.assert_called_once_with(file_id=1, user=self.mock_user)

    def test_download_file(self, test_client):
        """Test downloading a file."""
        # Configure mock service response
        file_content = b"col1,col2\n1,2\n3,4"
        self.mock_file_service.get_file_content.return_value = (
            file_content,
            "test.csv",
            "text/csv",
        )

        # Make the request
        response = test_client.get("/api/files/1/download")

        # Check response
        assert response.status_code == 200
        assert response.content == file_content
        assert response.headers["Content-Disposition"] == 'attachment; filename="test.csv"'
        assert response.headers["Content-Type"] == "text/csv"

        # Verify service was called
        self.mock_file_service.get_file_content.assert_called_once_with(
            file_id=1, user=self.mock_user
        )
