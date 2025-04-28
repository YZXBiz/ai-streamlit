"""Pytest configuration for tests."""

import os
from collections.abc import AsyncGenerator, Generator
from datetime import timedelta
from typing import Dict

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.app.adapters.db_postgres import (
    PostgresChatSessionRepository,
    PostgresDataFileRepository,
    PostgresUserRepository,
)
from backend.app.core.database.models import Base
from backend.app.core.database.session import get_db as orig_get_db
from backend.app.core.security import create_access_token
from backend.app.domain.models.chat_session import ChatSession
from backend.app.domain.models.datafile import DataFile, FileType
from backend.app.domain.models.user import User
from backend.app.main import app
from backend.app.services.auth_service import AuthService
from backend.app.services.file_service import FileService

# Use SQLite in-memory for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Setup test database
@pytest_asyncio.fixture
async def async_engine():
    """Create and return an async SQLite engine for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Get async session for testing."""
    async_session_maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def test_user() -> User:
    """Create a test user model."""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",  # "testpassword"
        first_name="Test",
        last_name="User",
        is_active=True,
        is_admin=False,
    )


@pytest_asyncio.fixture
async def test_admin_user() -> User:
    """Create a test admin user model."""
    return User(
        id=2,
        username="admin",
        email="admin@example.com",
        hashed_password="$2b$12$IW3GtL5YOolaTk8/s9.Ct.yIhgIzxj5pX/JPSHBjYwpQk5zUV.YhO",  # "testpassword"
        first_name="Admin",
        last_name="User",
        is_active=True,
        is_admin=True,
    )


@pytest_asyncio.fixture
async def test_data_file() -> DataFile:
    """Create a test data file model."""
    return DataFile(
        id=1,
        user_id=1,
        filename="test_data.csv",
        original_filename="sales_data.csv",
        file_path="/tmp/test_data.csv",
        file_size=1024,
        file_type=FileType.CSV,
        description="Test data file",
    )


@pytest_asyncio.fixture
async def test_chat_session() -> ChatSession:
    """Create a test chat session model."""
    return ChatSession(
        id=1,
        user_id=1,
        name="Test Session",
        description="Test chat session",
        data_file_id=1,
    )


@pytest_asyncio.fixture
async def test_user_in_db(async_session, test_user) -> User:
    """Add a test user to the database."""
    user_repo = PostgresUserRepository(async_session)
    return await user_repo.create(test_user)


@pytest_asyncio.fixture
async def test_admin_in_db(async_session, test_admin_user) -> User:
    """Add a test admin user to the database."""
    user_repo = PostgresUserRepository(async_session)
    return await user_repo.create(test_admin_user)


@pytest_asyncio.fixture
async def test_data_file_in_db(async_session, test_data_file, test_user_in_db) -> DataFile:
    """Add a test data file to the database."""
    file_repo = PostgresDataFileRepository(async_session)
    return await file_repo.create(test_data_file)


@pytest_asyncio.fixture
async def test_chat_session_in_db(
    async_session, test_chat_session, test_user_in_db, test_data_file_in_db
) -> ChatSession:
    """Add a test chat session to the database."""
    session_repo = PostgresChatSessionRepository(async_session)
    return await session_repo.create(test_chat_session)


# Repository fixtures
@pytest_asyncio.fixture
async def user_repository(async_session) -> PostgresUserRepository:
    """Get a user repository with test session."""
    return PostgresUserRepository(async_session)


@pytest_asyncio.fixture
async def file_repository(async_session) -> PostgresDataFileRepository:
    """Get a data file repository with test session."""
    return PostgresDataFileRepository(async_session)


@pytest_asyncio.fixture
async def chat_repository(async_session) -> PostgresChatSessionRepository:
    """Get a chat session repository with test session."""
    return PostgresChatSessionRepository(async_session)


# Service fixtures
@pytest_asyncio.fixture
async def auth_service(user_repository) -> AuthService:
    """Get an auth service with test repositories."""
    return AuthService(user_repository)


@pytest_asyncio.fixture
async def file_service(file_repository) -> FileService:
    """Get a file service with test repositories."""
    os.makedirs("./test_data", exist_ok=True)
    storage_path = "./test_data"
    return FileService(file_repository, storage_path)


# FastAPI test client fixture
@pytest_asyncio.fixture
async def client() -> Generator:
    """Get FastAPI test client."""

    # Override the get_db dependency
    async def override_get_db():
        engine = create_async_engine(TEST_DATABASE_URL)
        async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session_maker() as session:
            yield session
            await session.close()
        await engine.dispose()

    app.dependency_overrides[orig_get_db] = override_get_db

    # Create test tables
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    with TestClient(app) as client:
        yield client

    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def auth_headers(test_user_in_db) -> dict[str, str]:
    """Get authentication headers for test user."""
    access_token = create_access_token(
        subject=str(test_user_in_db.id),
        expires_delta=timedelta(minutes=30),
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest_asyncio.fixture
async def admin_headers(test_admin_in_db) -> dict[str, str]:
    """Get authentication headers for admin user."""
    access_token = create_access_token(
        subject=str(test_admin_in_db.id),
        expires_delta=timedelta(minutes=30),
    )
    return {"Authorization": f"Bearer {access_token}"}
