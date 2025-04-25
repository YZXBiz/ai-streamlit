"""
Common test fixtures for chatbot tests.

This module contains fixtures and configuration for pytest test cases.
"""

import os
import sys
import tempfile
import types
from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st
from _pytest.monkeypatch import MonkeyPatch
from dotenv import load_dotenv
from llama_index.core.schema import Document
from pandas import DataFrame

from chatbot.services.duckdb_base_service import DuckDBService
from chatbot.services.duckdb_enhanced_service import EnhancedDuckDBService

# from chatbot.services.duckdb_enhanced_service import EnhancedDuckDBService


# Set environment variables for testing
@pytest.fixture(autouse=True)
def env() -> None:
    """Set environment variables for testing."""
    # Save original environ
    old_environ = os.environ.copy()

    # Set testing variables
    os.environ["IS_TESTING"] = "true"
    os.environ["OPENAI_API_KEY"] = "sk-test-api-key-for-testing-only"
    os.environ["DB_PATH"] = ":memory:"
    os.environ["MEMORY_TYPE"] = "simple"
    os.environ["MEMORY_TOKEN_LIMIT"] = "1000"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    # Restore original environ
    os.environ.clear()
    os.environ.update(old_environ)


# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# Mock Streamlit
# -----------------------------------------------------------------------------
class MockSt:
    """Mock class for streamlit."""

    def __init__(self) -> None:
        """Initialize with mocked methods."""
        # Basic UI elements
        self.markdown = MagicMock()
        self.write = MagicMock()
        self.code = MagicMock()
        self.text_area = MagicMock()
        self.text_input = MagicMock(return_value="")
        self.selectbox = MagicMock()
        self.multiselect = MagicMock()
        self.checkbox = MagicMock(return_value=False)
        self.radio = MagicMock()
        self.button = MagicMock(return_value=False)
        self.form_submit_button = MagicMock(return_value=False)
        self.download_button = MagicMock()
        self.file_uploader = MagicMock(return_value=None)

        # Containers
        self.container = MagicMock()
        self.expander = MagicMock()
        self.spinner = MagicMock()
        self.form = MagicMock()

        # Setup container context managers
        self._setup_container_mock("container")
        self._setup_container_mock("expander")
        self._setup_container_mock("spinner")
        self._setup_container_mock("form")

        self.sidebar = MagicMock()
        # Make sidebar behave like a container
        self._setup_container_mock("sidebar")
        self.sidebar.button = MagicMock(return_value=False)
        self.sidebar.selectbox = MagicMock()
        self.sidebar.checkbox = MagicMock(return_value=False)
        self.sidebar.markdown = MagicMock()
        self.sidebar.text_input = MagicMock(return_value="")

        # Layout methods
        self.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        self.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Dataframe/data display methods
        self.dataframe = MagicMock()
        self.table = MagicMock()
        self.json = MagicMock()

        # Status messages
        self.info = MagicMock()
        self.success = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()

        # Control flow
        self.stop = MagicMock()
        self.rerun = MagicMock()

        # Settings
        self.set_page_config = MagicMock()

        # Session state
        self.session_state = {}

        # Other
        self.cache_resource = lambda func: func
        self.cache_data = lambda func: func
        self.experimental_rerun = MagicMock()

    def _setup_container_mock(self, name: str) -> None:
        """Set up a container mock with context manager functionality."""
        attr = getattr(self, name)
        mock_container = MagicMock()
        attr.return_value.__enter__ = MagicMock(return_value=mock_container)
        attr.return_value.__exit__ = MagicMock(return_value=None)

        # Add streamlit methods to containers
        mock_container.markdown = MagicMock()
        mock_container.write = MagicMock()
        mock_container.dataframe = MagicMock()
        mock_container.code = MagicMock()
        mock_container.info = MagicMock()
        mock_container.error = MagicMock()
        mock_container.warning = MagicMock()
        mock_container.success = MagicMock()


@pytest.fixture(autouse=True)
def mock_streamlit() -> Generator[MockSt, None, None]:
    """Mock streamlit module for all tests."""
    mock_st = MockSt()

    # Create a proper module-like object
    mock_module = types.ModuleType("streamlit")
    for attr_name in dir(mock_st):
        if not attr_name.startswith("_") or attr_name == "__enter__" or attr_name == "__exit__":
            setattr(mock_module, attr_name, getattr(mock_st, attr_name))

    with patch.dict(sys.modules, {"streamlit": mock_module}):
        yield mock_st


# -----------------------------------------------------------------------------
# Data Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_df() -> DataFrame:
    """Return a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        }
    )


@pytest.fixture
def sample_df2() -> DataFrame:
    """Return another sample DataFrame for testing joins."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 6, 7],
            "department": ["HR", "Engineering", "Marketing", "Finance", "Sales"],
            "salary": [50000, 80000, 60000, 70000, 55000],
        }
    )


# -----------------------------------------------------------------------------
# Database Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_db.duckdb")


@pytest.fixture
def duckdb_service(temp_db_path: str) -> Generator[DuckDBService, None, None]:
    """Return a DuckDBService instance with a temporary database."""
    service = DuckDBService(db_path=temp_db_path)
    yield service
    # Cleanup
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except PermissionError:
            pass  # It's OK if we can't remove it


@pytest.fixture
def loaded_duckdb_service(duckdb_service: DuckDBService, sample_df: DataFrame) -> DuckDBService:
    """Return a DuckDBService with data preloaded."""
    duckdb_service.load_dataframe(sample_df, "users")
    return duckdb_service


# -----------------------------------------------------------------------------
# Mock LlamaIndex Components
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_chat_memory() -> MagicMock:
    """Return a mock chat memory."""
    mock_memory = MagicMock()
    mock_memory.get_messages.return_value = []
    mock_memory.get.return_value = "No chat history available."
    return mock_memory


@pytest.fixture
def mock_embed_model() -> Any:
    """Return a mock OpenAI embedding model."""
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedModel(BaseEmbedding):
        """Mock implementation of OpenAI embedding model."""

        def _get_text_embedding(self, text: str) -> list[float]:
            """Return mock embeddings."""
            return [0.1 * i for i in range(10)]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            """Return mock embeddings."""
            return [[0.1 * i for i in range(10)] for _ in texts]

    return MockEmbedModel()


# -----------------------------------------------------------------------------
# Enhanced DuckDB Service Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def enhanced_duckdb_service(
    temp_db_path: str, mock_embed_model: Any, mock_chat_memory: MagicMock
) -> Generator[EnhancedDuckDBService, None, None]:
    """Return an EnhancedDuckDBService with mock embedding model."""
    with patch("chatbot.services.duckdb_enhanced_service.ChatMemoryBuffer") as mock_memory_class:
        mock_memory_class.from_defaults.return_value = mock_chat_memory

        service = EnhancedDuckDBService(
            embed_model=mock_embed_model,
            db_path=temp_db_path,
            memory_type="simple",
            token_limit=1000,
        )
        yield service
        # Cleanup
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except PermissionError:
                pass


@pytest.fixture
def loaded_enhanced_service(
    enhanced_duckdb_service: EnhancedDuckDBService,
    sample_df: DataFrame,
    sample_df2: DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> EnhancedDuckDBService:
    """Return an EnhancedDuckDBService with data preloaded."""
    # Mock VectorStoreIndex to avoid embedding calls
    with patch("chatbot.services.duckdb_enhanced_service.VectorStoreIndex") as mock_index_class:
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index

        # Mock retriever
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        # Mock SQL database
        with patch("chatbot.services.duckdb_enhanced_service.SQLDatabase") as mock_sql_db_class:
            mock_sql_db = MagicMock()
            mock_sql_db_class.return_value = mock_sql_db

            # Mock query engines
            with patch(
                "chatbot.services.duckdb_enhanced_service.NLSQLTableQueryEngine"
            ) as mock_simple_engine:
                with patch(
                    "chatbot.services.duckdb_enhanced_service.SQLTableRetrieverQueryEngine"
                ) as mock_advanced_engine:
                    simple_engine = MagicMock()
                    advanced_engine = MagicMock()
                    mock_simple_engine.return_value = simple_engine
                    mock_advanced_engine.return_value = advanced_engine

                    # Load data
                    enhanced_duckdb_service.load_dataframe(sample_df, "users")
                    enhanced_duckdb_service.load_dataframe(sample_df2, "departments")
                    enhanced_duckdb_service.initialize()

                    # Verify query engines were set up
                    assert enhanced_duckdb_service.simple_query_engine is not None
                    assert enhanced_duckdb_service.advanced_query_engine is not None

                    yield enhanced_duckdb_service


# -----------------------------------------------------------------------------
# Mock LlamaIndex Components
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_chat_memory() -> MagicMock:
    """Return a mock chat memory."""
    mock_memory = MagicMock()
    mock_memory.get_messages.return_value = []
    mock_memory.get.return_value = "No chat history available."
    return mock_memory


@pytest.fixture
def mock_embed_model() -> Any:
    """Return a mock OpenAI embedding model."""
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedModel(BaseEmbedding):
        """Mock implementation of OpenAI embedding model."""

        def _get_text_embedding(self, text: str) -> list[float]:
            """Return mock embeddings."""
            return [0.1 * i for i in range(10)]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            """Return mock embeddings."""
            return [[0.1 * i for i in range(10)] for _ in texts]

    return MockEmbedModel()


# -----------------------------------------------------------------------------
# Enhanced DuckDB Service Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock OpenAI API key in environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-for-testing")
    """Set mock OpenAI API key in environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-for-testing")
