"""
Security utilities for authentication and authorization.

This module provides functions for password hashing, token generation,
and verification for the authentication system.
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict

from .config import settings

# Define password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token URL - update this to match your actual token endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class TokenData(BaseModel):
    """Token data model for JWT payload."""

    username: str
    user_id: int
    exp: datetime


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify that the provided plain password matches the hashed password.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to check against

    Returns:
        bool: True if the password is valid, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password with bcrypt.

    Args:
        password: The plain text password to hash

    Returns:
        str: The hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time delta, defaults to settings.ACCESS_TOKEN_EXPIRE_MINUTES

    Returns:
        str: JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    Decode a JWT token and extract the TokenData.

    Args:
        token: The JWT token to decode

    Returns:
        TokenData: The decoded token data

    Raises:
        HTTPException: If the token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Verify token and decode payload
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

        # Extract username and optionally user_id
        username: str = payload.get("sub")
        exp: int = payload.get("exp")
        user_id: int | None = payload.get("user_id")

        if username is None:
            raise credentials_exception

        return TokenData(username=username, user_id=user_id, exp=exp)
    except JWTError as e:
        raise credentials_exception from e


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Get the current user from the JWT token.

    This dependency will automatically extract the token from the Authorization header,
    decode it, and return the token data if valid.

    Args:
        token: The JWT token from the Authorization header (injected by FastAPI)

    Returns:
        TokenData: The decoded token data including username and user_id

    Raises:
        HTTPException: If the token is invalid or expired
    """
    return decode_token(token)
