"""Tests for the vector store implementation."""

import os
import shutil
from collections.abc import AsyncGenerator

import pytest

from backend.app.adapters.vector_simple import SimpleTfidfVectorStore
from backend.app.ports.vectorstore import VectorStore


@pytest.fixture
async def vector_store() -> AsyncGenerator[VectorStore, None]:
    """Create a test vector store."""
    # Create a temporary directory for testing
    os.makedirs("./data/test_vector_indices", exist_ok=True)

    # Set up the vector store
    store = SimpleTfidfVectorStore(storage_dir="./data/test_vector_indices")

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

    # Check that the session was created
    assert 1 in vector_store.sessions
    assert len(vector_store.sessions[1]["texts"]) == 1
    assert len(vector_store.sessions[1]["metadata"]) == 1

    # Check that the memory has the correct text
    assert vector_store.sessions[1]["texts"][0] == "This is a test memory"

    # Check that the metadata was stored
    assert vector_store.sessions[1]["metadata"][0]["test"] == "value"
    assert vector_store.sessions[1]["metadata"][0]["id"] == memory_id


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

    # Check that the results are about programming languages
    texts = [result["text"] for result in results]
    assert "Python is a programming language" in texts
    assert "Java is also a programming language" in texts

    # Query for animals
    results = await vector_store.query_memory(
        session_id=1, query_text="Tell me about animals", top_k=2
    )

    # Check that we got results
    assert len(results) == 2

    # Check that the results are about animals
    texts = [result["text"] for result in results]
    assert "Cats are animals" in texts
    assert "Dogs are also animals" in texts


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
