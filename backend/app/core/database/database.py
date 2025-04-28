"""
Database core components.

This module defines the SQLAlchemy Base class and session utilities.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..config import settings

# Base class for SQLAlchemy models
Base = declarative_base()

# Create async PostgreSQL engine
async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=settings.SQL_ECHO,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
)

# Session factory for async operations
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncSession:
    """
    Get an async database session.

    Yields:
        AsyncSession: Database session

    Usage:
        ```
        async with get_async_session() as session:
            # use session
        ```
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Create synchronous PostgreSQL engine (for migrations and sync operations if needed)
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.SQL_ECHO,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
)

# Session factory for synchronous operations
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


def get_sync_session():
    """
    Get a synchronous database session.

    Yields:
        Session: Database session

    Usage:
        ```
        with get_sync_session() as session:
            # use session
        ```
    """
    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.close()
