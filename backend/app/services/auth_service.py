"""
Authentication service for user management.

This service handles user authentication, registration, and token management.
"""

from datetime import timedelta

from fastapi import HTTPException, status

from ..core.config import settings
from ..core.security import create_access_token, get_password_hash, verify_password
from ..domain.models.user import User
from ..ports.repository import UserRepository


class AuthService:
    """Service for user authentication."""

    def __init__(self, user_repository: UserRepository):
        """
        Initialize with a user repository.

        Args:
            user_repository: Repository for user operations
        """
        self.user_repository = user_repository

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """
        Authenticate a user with username/password.

        Args:
            username: Username
            password: Plain password

        Returns:
            User if authentication succeeds, None otherwise
        """
        user = await self.user_repository.get_by_username(username)
        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    async def create_user(
        self, username: str, email: str, password: str, is_admin: bool = False
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            password: Plain password
            is_admin: Whether the user is an admin

        Returns:
            Created user

        Raises:
            HTTPException: If username or email already exists
        """
        # Check if username already exists
        existing_user = await self.user_repository.get_by_username(username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        # Check if email already exists
        existing_email = await self.user_repository.get_by_email(email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create new user
        hashed_password = get_password_hash(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_admin=is_admin,
        )

        # Save user to repository
        return await self.user_repository.create(user)

    async def login(self, username: str, password: str) -> tuple[User, str]:
        """
        Log in a user and generate an access token.

        Args:
            username: Username
            password: Plain password

        Returns:
            Tuple of (user, access_token)

        Raises:
            HTTPException: If authentication fails
        """
        user = await self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id}, expires_delta=access_token_expires
        )

        return user, access_token

    async def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """
        Change a user's password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            HTTPException: If user not found or current password is incorrect
        """
        user = await self.user_repository.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

        # Update password
        user.hashed_password = get_password_hash(new_password)
        await self.user_repository.update(user)

        return True
