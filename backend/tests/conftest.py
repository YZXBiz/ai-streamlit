"""Pytest configuration for tests."""

import os
import sys
from collections.abc import AsyncGenerator, Generator
from datetime import timedelta
from typing import Dict
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Set environment variables
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "1440"
os.environ["DB_PORT"] = "5432"
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

# Mock the settings module first
settings_mock = MagicMock()
settings_mock.API_V1_STR = "/api/v1"
settings_mock.SECRET_KEY = "test-secret"
settings_mock.ACCESS_TOKEN_EXPIRE_MINUTES = 30
settings_mock.DB_HOST = "localhost"
settings_mock.DB_PORT = 5432
settings_mock.DB_USER = "postgres"
settings_mock.DB_PASSWORD = "postgres"
settings_mock.DB_NAME = "test_db"
settings_mock.OPENAI_API_KEY = "sk-dummy-key-for-testing"
settings_mock.DATA_DIR = "./data"
settings_mock.LOGS_DIR = "./logs"
settings_mock.STORAGE_PATH = "./data/uploads"
settings_mock.VECTOR_STORE_DIR = "./data/vector_store"

# Mock modules to bypass importing issues
sys.modules["backend.app.core.config"] = MagicMock()
sys.modules["backend.app.core.config"].settings = settings_mock

sys.modules["backend.app.adapters.db_postgres"] = MagicMock()
sys.modules["backend.app.core.database.models"] = MagicMock()
sys.modules["backend.app.core.database.session"] = MagicMock()
sys.modules["backend.app.core.database.database"] = MagicMock()
sys.modules["backend.app.core.security"] = MagicMock()
sys.modules["backend.app.domain.models.chat_session"] = MagicMock()
sys.modules["backend.app.domain.models.datafile"] = MagicMock()
sys.modules["backend.app.domain.models.user"] = MagicMock()
sys.modules["backend.app.services.auth_service"] = MagicMock()
sys.modules["backend.app.services.file_service"] = MagicMock()
sys.modules["backend.app.services.chat_service"] = MagicMock()
sys.modules["backend.app.services.analyzer_service"] = MagicMock()
sys.modules["backend.app.ports.llm"] = MagicMock()
sys.modules["backend.app.ports.repository"] = MagicMock()
sys.modules["backend.app.ports.storage"] = MagicMock()
sys.modules["backend.app.ports.vectorstore"] = MagicMock()
sys.modules["backend.app.adapters.llm_pandasai"] = MagicMock()
sys.modules["backend.app.adapters.storage_local"] = MagicMock()
sys.modules["backend.app.adapters.vector_faiss"] = MagicMock()
sys.modules["backend.app.adapters.sandbox"] = MagicMock()

# Setup service mocks with async methods
auth_service_mock = AsyncMock()
auth_service_mock.login.return_value = (MagicMock(id=1, username="testuser"), "test-token")
auth_service_mock.create_user.return_value = MagicMock(
    id=1,
    username="testuser",
    email="test@example.com",
    first_name="Test",
    last_name="User",
    is_active=True,
    is_admin=False,
    created_at=None,
)

file_service_mock = AsyncMock()
file_service_mock.save_file.return_value = MagicMock(id=1, filename="test.csv")
file_service_mock.get_file.return_value = MagicMock(id=1, filename="test.csv")

# Configure dependencies
sys.modules["backend.app.api.deps"] = MagicMock()
sys.modules["backend.app.api.deps"].get_auth_service = AsyncMock(return_value=auth_service_mock)
sys.modules["backend.app.api.deps"].get_file_service = AsyncMock(return_value=file_service_mock)
sys.modules["backend.app.api.deps"].get_chat_service = AsyncMock()
sys.modules["backend.app.api.deps"].get_analyzer_service = AsyncMock()
sys.modules["backend.app.api.deps"].get_current_user = AsyncMock(return_value=MagicMock(user_id=1))
sys.modules["backend.app.api.deps"].get_current_active_user = AsyncMock(
    return_value=MagicMock(user_id=1)
)


# Simple test client fixture
@pytest.fixture
def client():
    """Get a test client for the FastAPI app."""
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client
