"""Tests for VectorStore port."""

import pytest
from app.ports.vectorstore import VectorStore


def test_vectorstore_is_abstract():
    """Test that VectorStore is an abstract base class."""
    # Trying to instantiate VectorStore should raise TypeError
    with pytest.raises(TypeError):
        VectorStore()


def test_vectorstore_abstract_methods():
    """Test that VectorStore has the expected abstract methods."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(VectorStore)
        if getattr(getattr(VectorStore, method_name), "__isabstractmethod__", False)
    ]

    # Check that all the expected methods are abstract
    expected_methods = [
        "add_memory",
        "query_memory",
        "delete_session_memories",
        "get_session_memories",
    ]
    for method in expected_methods:
        assert method in abstract_methods


class ConcreteVectorStoreForTesting(VectorStore):
    """A concrete implementation of VectorStore for testing."""

    async def add_memory(self, session_id, text, metadata=None):
        return "test-id"

    async def query_memory(self, session_id, query_text, top_k=5):
        return []

    async def delete_session_memories(self, session_id):
        return True

    async def get_session_memories(self, session_id, limit=100):
        return []


def test_concrete_vectorstore():
    """Test that a concrete implementation of VectorStore can be instantiated."""
    store = ConcreteVectorStoreForTesting()
    assert isinstance(store, VectorStore)
