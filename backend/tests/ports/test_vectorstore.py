"""Tests for vectorstore module."""

import inspect
from abc import ABC
from typing import Any

import pytest


def test_vectorstore_constants():
    """Test vectorstore module constants."""
    from app.ports.vectorstore import VectorStore

    assert hasattr(VectorStore, "__abstractmethods__")


class TestVectorStore:
    """Tests for the VectorStore interface."""

    @pytest.mark.port
    def test_vector_store_is_abc(self):
        """Test that VectorStore is an ABC."""
        from app.ports.vectorstore import VectorStore

        assert issubclass(VectorStore, ABC)

    @pytest.mark.port
    def test_vector_store_methods(self):
        """Test that VectorStore has the expected methods."""
        from app.ports.vectorstore import VectorStore

        # Check required methods
        assert hasattr(VectorStore, "add_memory")
        assert hasattr(VectorStore, "query_memory")
        assert hasattr(VectorStore, "delete_session_memories")
        assert hasattr(VectorStore, "get_session_memories")

    @pytest.mark.port
    def test_add_memory_signature(self):
        """Test that add_memory has the expected signature."""
        from app.ports.vectorstore import VectorStore

        sig = inspect.signature(VectorStore.add_memory)
        params = sig.parameters

        assert "self" in params
        assert "session_id" in params
        assert "text" in params
        assert "metadata" in params

        # Check parameter types and defaults
        assert params["session_id"].annotation == int
        assert params["text"].annotation == str
        assert params["metadata"].annotation == "dict[str, Any] | None"
        assert params["metadata"].default is None

        # Check return type
        assert sig.return_annotation == str

    @pytest.mark.port
    def test_query_memory_signature(self):
        """Test that query_memory has the expected signature."""
        from app.ports.vectorstore import VectorStore

        sig = inspect.signature(VectorStore.query_memory)
        params = sig.parameters

        assert "self" in params
        assert "session_id" in params
        assert "query_text" in params
        assert "top_k" in params

        # Check parameter types and defaults
        assert params["session_id"].annotation == int
        assert params["query_text"].annotation == str
        assert params["top_k"].annotation == int
        assert params["top_k"].default == 5

        # Check return type
        assert sig.return_annotation == "list[dict[str, Any]]"

    @pytest.mark.port
    def test_delete_session_memories_signature(self):
        """Test that delete_session_memories has the expected signature."""
        from app.ports.vectorstore import VectorStore

        sig = inspect.signature(VectorStore.delete_session_memories)
        params = sig.parameters

        assert "self" in params
        assert "session_id" in params

        # Check parameter types
        assert params["session_id"].annotation == int

        # Check return type
        assert sig.return_annotation == bool

    @pytest.mark.port
    def test_get_session_memories_signature(self):
        """Test that get_session_memories has the expected signature."""
        from app.ports.vectorstore import VectorStore

        sig = inspect.signature(VectorStore.get_session_memories)
        params = sig.parameters

        assert "self" in params
        assert "session_id" in params
        assert "limit" in params

        # Check parameter types and defaults
        assert params["session_id"].annotation == int
        assert params["limit"].annotation == int
        assert params["limit"].default == 100

        # Check return type
        assert sig.return_annotation == "list[dict[str, Any]]"
