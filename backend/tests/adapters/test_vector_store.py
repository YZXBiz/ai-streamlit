"""Tests for the vector store implementation."""

import os
import shutil
from collections.abc import AsyncGenerator

import pytest
from backend.app.adapters.vector_faiss import FAISSVectorStore
from backend.app.ports.vectorstore import VectorStore


@pytest.fixture
async def vector_store() -> AsyncGenerator[VectorStore, None]:
    """Create a test vector store."""
    # Create a temporary directory for testing
    os.makedirs("./data/test_vector_indices", exist_ok=True)

    # Set up the vector store
    store = FAISSVectorStore(model_name="all-MiniLM-L6-v2")

    yield store

    # Clean up
    shutil.rmtree("./data/test_vector_indices", ignore_errors=True)


@pytest.mark.asyncio
async def test_add_memory(vector_store: VectorStore) -> None:
    """Test adding a memory to the vector store."""
    # Add a memory
    memory_id = await vector_store.add_memory(
        session_id=1, text="This is a test memory", metadata={"test": "value"}
    )

    # Check that the memory was added
    assert memory_id is not None
    assert isinstance(memory_id, str)

    # Get all memories to verify the addition
    memories = await vector_store.get_session_memories(session_id=1)
    assert len(memories) == 1

    # Verify that our memory exists with correct metadata
    memory = memories[0]
    assert memory["text"] == "This is a test memory"
    assert memory["test"] == "value"


@pytest.mark.asyncio
async def test_query_memory(vector_store: VectorStore) -> None:
    """Test querying memories from the vector store."""
    # Add some memories
    await vector_store.add_memory(session_id=1, text="Python is a programming language")
    await vector_store.add_memory(session_id=1, text="Java is also a programming language")
    await vector_store.add_memory(session_id=1, text="Cats are animals")
    await vector_store.add_memory(session_id=1, text="Dogs are also animals")

    # Query for programming languages
    results = await vector_store.query_memory(
        session_id=1, query_text="Tell me about programming languages", top_k=2
    )

    # Check that we got results
    assert len(results) == 2

    # Check that the results are relevant to programming languages
    programming_terms = ["programming", "language", "Python", "Java"]
    for result in results:
        assert any(term.lower() in result["text"].lower() for term in programming_terms)

    # Query for animals
    results = await vector_store.query_memory(
        session_id=1, query_text="Tell me about animals", top_k=2
    )

    # Check that we got results
    assert len(results) == 2

    # Check that the results are relevant to animals
    animal_terms = ["animal", "cat", "dog"]
    for result in results:
        assert any(term.lower() in result["text"].lower() for term in animal_terms)


@pytest.mark.asyncio
async def test_delete_session_memories(vector_store: VectorStore) -> None:
    """Test deleting memories for a session."""
    # Add some memories
    await vector_store.add_memory(session_id=1, text="Memory 1")
    await vector_store.add_memory(session_id=1, text="Memory 2")

    # Delete the session
    result = await vector_store.delete_session_memories(session_id=1)

    # Check that the deletion was successful
    assert result is True

    # Check that the session was deleted
    assert 1 not in vector_store.sessions

    # Check that querying the deleted session returns an empty list
    results = await vector_store.query_memory(session_id=1, query_text="test")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_session_memories(vector_store: VectorStore) -> None:
    """Test getting all memories for a session."""
    # Add some memories
    await vector_store.add_memory(session_id=1, text="Memory 1")
    await vector_store.add_memory(session_id=1, text="Memory 2")
    await vector_store.add_memory(session_id=1, text="Memory 3")

    # Get all memories
    memories = await vector_store.get_session_memories(session_id=1)

    # Check that we got all memories
    assert len(memories) == 3

    # Check that the memories have the correct texts
    texts = [memory["text"] for memory in memories]
    assert "Memory 1" in texts
    assert "Memory 2" in texts
    assert "Memory 3" in texts

    # Get limited memories
    memories = await vector_store.get_session_memories(session_id=1, limit=2)

    # Check that we got the limited number of memories
    assert len(memories) == 2
