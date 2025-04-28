"""
Database session management.

This module handles SQLAlchemy session creation and lifecycle management.
"""

from collections.abc import Generator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings
from .models import Base

# Create async engine
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> Generator[AsyncSession, None, None]:
    """
    Get a database session.

    Yields:
        AsyncSession: SQLAlchemy async session

    Example:
        ```python
        async with get_db() as db:
            # Use db session
            result = await db.execute(...)
        ```
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables() -> None:
    """Create all database tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
