"""Tests for the authentication service."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from backend.app.domain.models.user import User
from backend.app.ports.repository import UserRepository
from backend.app.services.auth_service import AuthService

from backend.app.core.security import verify_password


@pytest.fixture
def mock_user_repo():
    """Create a mock user repository."""
    repo = MagicMock(spec=UserRepository)

    # Setup the get_by_username method to return a User
    repo.get_by_username.return_value = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",  # hashed "testpassword"
        first_name="Test",
        last_name="User",
        is_active=True,
        is_admin=False,
    )

    # Setup the get_by_email method to return a User
    repo.get_by_email.return_value = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",  # hashed "testpassword"
        first_name="Test",
        last_name="User",
        is_active=True,
        is_admin=False,
    )

    # Setup the create method to return a User with assigned ID
    repo.create.side_effect = lambda user: User(
        id=1,
        username=user.username,
        email=user.email,
        hashed_password=user.hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
    )

    return repo


class TestAuthService:
    """Test the AuthService."""

    @pytest.mark.service
    def test_init(self, mock_user_repo):
        """Test initialization of AuthService."""
        service = AuthService(user_repository=mock_user_repo)
        assert service.user_repository == mock_user_repo

    @pytest.mark.service
    async def test_authenticate_user_success(self, mock_user_repo):
        """Test successful user authentication."""
        service = AuthService(user_repository=mock_user_repo)

        # The user repository will return a user with the hashed version of "testpassword"
        user = await service.authenticate_user(username="testuser", password="testpassword")

        assert user is not None
        assert user.id == 1
        assert user.username == "testuser"

        # Verify repository was called
        mock_user_repo.get_by_username.assert_called_once_with("testuser")

    @pytest.mark.service
    async def test_authenticate_user_wrong_password(self, mock_user_repo):
        """Test authentication with wrong password."""
        service = AuthService(user_repository=mock_user_repo)

        user = await service.authenticate_user(username="testuser", password="wrongpassword")

        assert user is None

        # Verify repository was called
        mock_user_repo.get_by_username.assert_called_once_with("testuser")

    @pytest.mark.service
    async def test_authenticate_user_nonexistent(self, mock_user_repo):
        """Test authentication with nonexistent user."""
        # Set up mock to return None for nonexistent user
        mock_user_repo.get_by_username.return_value = None

        service = AuthService(user_repository=mock_user_repo)

        user = await service.authenticate_user(username="nonexistent", password="testpassword")

        assert user is None

        # Verify repository was called
        mock_user_repo.get_by_username.assert_called_once_with("nonexistent")

    @pytest.mark.service
    @patch("app.services.auth_service.get_password_hash")
    async def test_register_user(self, mock_get_password_hash, mock_user_repo):
        """Test user registration."""
        # Mock password hashing
        mock_get_password_hash.return_value = "hashed_password"

        # Set up mock to return None for get_by_username and get_by_email (user doesn't exist)
        mock_user_repo.get_by_username.return_value = None
        mock_user_repo.get_by_email.return_value = None

        service = AuthService(user_repository=mock_user_repo)

        user = await service.register_user(
            username="newuser",
            email="new@example.com",
            password="password123",
            first_name="New",
            last_name="User",
        )

        assert user is not None
        assert user.id == 1
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.hashed_password == "hashed_password"
        assert user.first_name == "New"
        assert user.last_name == "User"
        assert user.is_active is True
        assert user.is_admin is False

        # Verify repository was called
        mock_user_repo.get_by_username.assert_called_once_with("newuser")
        mock_user_repo.get_by_email.assert_called_once_with("new@example.com")
        mock_user_repo.create.assert_called_once()

        # Verify password was hashed
        mock_get_password_hash.assert_called_once_with("password123")

    @pytest.mark.service
    async def test_register_user_existing_username(self, mock_user_repo):
        """Test registration with existing username."""
        # Mock user repository to return a user (username exists)
        mock_user_repo.get_by_username.return_value = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            first_name="Test",
            last_name="User",
            is_active=True,
            is_admin=False,
        )

        service = AuthService(user_repository=mock_user_repo)

        with pytest.raises(ValueError, match="Username already registered"):
            await service.register_user(
                username="testuser",
                email="new@example.com",
                password="password123",
                first_name="New",
                last_name="User",
            )

        # Verify repository was called for username check
        mock_user_repo.get_by_username.assert_called_once_with("testuser")

        # Verify create was not called
        mock_user_repo.create.assert_not_called()

    @pytest.mark.service
    async def test_register_user_existing_email(self, mock_user_repo):
        """Test registration with existing email."""
        # Set up mock to return None for username check (username doesn't exist)
        mock_user_repo.get_by_username.return_value = None

        # Set up mock to return a user for email check (email exists)
        mock_user_repo.get_by_email.return_value = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            first_name="Test",
            last_name="User",
            is_active=True,
            is_admin=False,
        )

        service = AuthService(user_repository=mock_user_repo)

        with pytest.raises(ValueError, match="Email already registered"):
            await service.register_user(
                username="newuser",
                email="test@example.com",
                password="password123",
                first_name="New",
                last_name="User",
            )

        # Verify repository was called for username and email check
        mock_user_repo.get_by_username.assert_called_once_with("newuser")
        mock_user_repo.get_by_email.assert_called_once_with("test@example.com")

        # Verify create was not called
        mock_user_repo.create.assert_not_called()

    @pytest.mark.service
    @patch("app.services.auth_service.create_access_token")
    async def test_login(self, mock_create_access_token, mock_user_repo):
        """Test user login."""
        # Mock token creation
        mock_create_access_token.return_value = "test_token"

        service = AuthService(user_repository=mock_user_repo)

        token = await service.login(username="testuser", password="testpassword")

        assert token == "test_token"

        # Verify token creation was called with correct user ID
        mock_create_access_token.assert_called_once()
        args, kwargs = mock_create_access_token.call_args
        assert kwargs["subject"] == "1"  # User ID as string

    @pytest.mark.service
    async def test_login_failure(self, mock_user_repo):
        """Test login failure."""
        service = AuthService(user_repository=mock_user_repo)

        token = await service.login(username="testuser", password="wrongpassword")

        assert token is None
