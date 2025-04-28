"""
Dependency injection functions for FastAPI endpoints.

This module provides functions that can be used as dependencies in FastAPI
route handlers to inject services and other dependencies.
"""

from fastapi import HTTPException, status

from ..adapters.db_postgres import (
    PostgresChatSessionRepository,
    PostgresDataFileRepository,
    PostgresUserRepository,
)
from ..adapters.llm_pandasai import PandasAiAdapter
from ..adapters.storage_local import LocalFileStorage
from ..adapters.vector_faiss import FAISSVectorStore
from ..core.config import settings
from ..core.database.database import get_async_session
from ..core.security import TokenData, get_current_user
from ..ports.llm import DataAnalysisService
from ..ports.repository import ChatSessionRepository, DataFileRepository, UserRepository
from ..ports.storage import FileStorage
from ..ports.vectorstore import VectorStore
from ..services.analyzer_service import AnalyzerService
from ..services.auth_service import AuthService
from ..services.chat_service import ChatService
from ..services.file_service import FileService


# Repository dependencies
async def get_user_repository() -> UserRepository:
    """Get user repository."""
    db = await anext(get_async_session())
    return PostgresUserRepository(db)


async def get_datafile_repository() -> DataFileRepository:
    """Get data file repository."""
    db = await anext(get_async_session())
    return PostgresDataFileRepository(db)


async def get_chatsession_repository() -> ChatSessionRepository:
    """Get chat session repository."""
    db = await anext(get_async_session())
    return PostgresChatSessionRepository(db)


# Storage and data service dependencies
async def get_file_storage() -> FileStorage:
    """Get file storage adapter."""
    return LocalFileStorage(settings.STORAGE_PATH)


async def get_data_service() -> DataAnalysisService:
    """Get data analysis service."""
    return PandasAiAdapter(settings.OPENAI_API_KEY)


async def get_vector_store() -> VectorStore:
    """Get vector store for long-term memory."""
    return FAISSVectorStore()


# Service dependencies
async def get_auth_service() -> AuthService:
    """Get authentication service."""
    user_repository = await get_user_repository()
    return AuthService(user_repository)


async def get_file_service() -> FileService:
    """Get file service."""
    file_repository = await get_datafile_repository()
    file_storage = await get_file_storage()
    data_service = await get_data_service()
    return FileService(file_repository, file_storage, data_service)


async def get_chat_service() -> ChatService:
    """Get chat service."""
    session_repository = await get_chatsession_repository()
    file_repository = await get_datafile_repository()
    data_service = await get_data_service()
    vector_store = await get_vector_store()
    return ChatService(session_repository, file_repository, data_service, vector_store)


async def get_analyzer_service() -> AnalyzerService:
    """Get analyzer service for data analysis."""
    data_service = await get_data_service()
    return AnalyzerService(data_service)


# User dependencies
async def get_current_active_user() -> TokenData:
    """Get current active user from token."""
    token_data = await get_current_user()
    user_repository = await get_user_repository()

    # Verify user exists and is active
    user = await user_repository.get(token_data.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive or non-existent user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


async def get_current_admin_user() -> TokenData:
    """Get current admin user from token."""
    token_data = await get_current_active_user()
    user_repository = await get_user_repository()

    # Verify user is admin
    user = await user_repository.get(token_data.user_id)
    if user is None or not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return token_data
