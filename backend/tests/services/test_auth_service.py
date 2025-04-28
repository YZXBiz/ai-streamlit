"""Tests for the auth service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.app.domain.models.user import User
from backend.app.services.auth_service import AuthService


@pytest.mark.asyncio
class TestAuthService:
    """Tests for the AuthService class."""

    async def test_verify_password(self):
        """Test password verification."""
        # Mock bcrypt.checkpw to return True
        with patch("backend.app.services.auth_service.pwd_context.verify") as mock_verify:
            mock_verify.return_value = True

            # Create a mock UserRepository
            user_repo = AsyncMock()
            auth_service = AuthService(user_repo)

            # Test with valid password
            result = auth_service.verify_password(
                "testpassword", "$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO"
            )
            assert result is True
            mock_verify.assert_called_once()

        # Mock bcrypt.checkpw to return False
        with patch("backend.app.services.auth_service.pwd_context.verify") as mock_verify:
            mock_verify.return_value = False

            # Create a mock UserRepository
            user_repo = AsyncMock()
            auth_service = AuthService(user_repo)

            # Test with invalid password
            result = auth_service.verify_password(
                "wrongpassword", "$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO"
            )
            assert result is False
            mock_verify.assert_called_once()

    async def test_get_password_hash(self):
        """Test password hashing."""
        with patch("backend.app.services.auth_service.pwd_context.hash") as mock_hash:
            mock_hash.return_value = "hashed_password"

            # Create a mock UserRepository
            user_repo = AsyncMock()
            auth_service = AuthService(user_repo)

            # Test hashing
            result = auth_service.get_password_hash("testpassword")
            assert result == "hashed_password"
            mock_hash.assert_called_once_with("testpassword")

    async def test_login_success(self):
        """Test successful login."""
        # Create a mock user
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = user

        # Create the auth service with mocked dependencies
        auth_service = AuthService(user_repo)

        # Mock password verification
        auth_service.verify_password = MagicMock(return_value=True)

        # Mock token creation
        with patch("backend.app.services.auth_service.create_access_token") as mock_create_token:
            mock_create_token.return_value = "mock_token"

            # Test login
            result_user, result_token = await auth_service.login("testuser", "testpassword")

            # Verify results
            assert result_user == user
            assert result_token == "mock_token"
            user_repo.get_by_username.assert_called_once_with("testuser")
            auth_service.verify_password.assert_called_once_with(
                "testpassword", user.hashed_password
            )
            mock_create_token.assert_called_once()

    async def test_login_user_not_found(self):
        """Test login with non-existent user."""
        # Create a mock UserRepository that returns None
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = None

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test login with non-existent user
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.login("nonexistent", "testpassword")

        # Verify exception
        assert excinfo.value.status_code == 401
        assert "Invalid username or password" in excinfo.value.detail
        user_repo.get_by_username.assert_called_once_with("nonexistent")

    async def test_login_inactive_user(self):
        """Test login with inactive user."""
        # Create a mock inactive user
        user = User(
            id=1,
            username="inactive",
            email="inactive@example.com",
            hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",
            is_active=False,
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = user

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test login with inactive user
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.login("inactive", "testpassword")

        # Verify exception
        assert excinfo.value.status_code == 401
        assert "Account is inactive" in excinfo.value.detail
        user_repo.get_by_username.assert_called_once_with("inactive")

    async def test_login_wrong_password(self):
        """Test login with wrong password."""
        # Create a mock user
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = user

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Mock password verification to return False
        auth_service.verify_password = MagicMock(return_value=False)

        # Test login with wrong password
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.login("testuser", "wrongpassword")

        # Verify exception
        assert excinfo.value.status_code == 401
        assert "Invalid username or password" in excinfo.value.detail
        user_repo.get_by_username.assert_called_once_with("testuser")
        auth_service.verify_password.assert_called_once_with("wrongpassword", user.hashed_password)

    async def test_create_user_success(self):
        """Test successful user creation."""
        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = None
        user_repo.get_by_email.return_value = None

        # Mock the create method to return a user with ID
        async def mock_create(user):
            user.id = 1
            return user

        user_repo.create.side_effect = mock_create

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Mock password hashing
        auth_service.get_password_hash = MagicMock(return_value="hashed_password")

        # Test user creation
        user = await auth_service.create_user(
            username="newuser",
            email="new@example.com",
            password="newpassword",
            first_name="New",
            last_name="User",
        )

        # Verify results
        assert user.id == 1
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.hashed_password == "hashed_password"
        assert user.first_name == "New"
        assert user.last_name == "User"
        assert user.is_active is True
        assert user.is_admin is False

        user_repo.get_by_username.assert_called_once_with("newuser")
        user_repo.get_by_email.assert_called_once_with("new@example.com")
        auth_service.get_password_hash.assert_called_once_with("newpassword")
        assert user_repo.create.call_count == 1

    async def test_create_user_username_exists(self):
        """Test user creation with existing username."""
        # Create a mock existing user
        existing_user = User(
            id=1,
            username="existing",
            email="old@example.com",
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = existing_user

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test user creation with existing username
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.create_user(
                username="existing",
                email="new@example.com",
                password="password",
            )

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Username already registered" in excinfo.value.detail
        user_repo.get_by_username.assert_called_once_with("existing")
        user_repo.get_by_email.assert_not_called()
        user_repo.create.assert_not_called()

    async def test_create_user_email_exists(self):
        """Test user creation with existing email."""
        # Create a mock existing user
        existing_user = User(
            id=1,
            username="olduser",
            email="existing@example.com",
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get_by_username.return_value = None
        user_repo.get_by_email.return_value = existing_user

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test user creation with existing email
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.create_user(
                username="newuser",
                email="existing@example.com",
                password="password",
            )

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Email already registered" in excinfo.value.detail
        user_repo.get_by_username.assert_called_once_with("newuser")
        user_repo.get_by_email.assert_called_once_with("existing@example.com")
        user_repo.create.assert_not_called()

    async def test_get_user_by_id_success(self):
        """Test getting a user by ID successfully."""
        # Create a mock user
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
        )

        # Create a mock UserRepository
        user_repo = AsyncMock()
        user_repo.get.return_value = user

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test getting user by ID
        result = await auth_service.get_user_by_id(1)

        # Verify results
        assert result == user
        user_repo.get.assert_called_once_with(1)

    async def test_get_user_by_id_not_found(self):
        """Test getting a non-existent user by ID."""
        # Create a mock UserRepository that returns None
        user_repo = AsyncMock()
        user_repo.get.return_value = None

        # Create the auth service
        auth_service = AuthService(user_repo)

        # Test getting non-existent user
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_user_by_id(999)

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "User not found" in excinfo.value.detail
        user_repo.get.assert_called_once_with(999)
