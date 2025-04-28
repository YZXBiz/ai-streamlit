"""Tests for FAISSVectorStore adapter."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock the Settings class before it's imported
mock_settings = MagicMock()
mock_settings.DATA_DIR = "./data"
mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"

with patch("app.core.config.settings", mock_settings):
    from app.adapters.vector_faiss import FAISSVectorStore

from app.domain.models.vector_store import VectorStoreDocument


@pytest.fixture
def vector_store_dir():
    """Create a temporary directory for vector store indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def faiss_store(vector_store_dir):
    """Create a FAISSVectorStore instance for testing."""
    with patch("os.makedirs"), patch.object(FAISSVectorStore, "_load_indices"):
        # Override index_dir in constructor
        store = FAISSVectorStore(model_name="all-MiniLM-L6-v2")
        store.index_dir = vector_store_dir
        return store


def test_faiss_store_initialization(faiss_store, vector_store_dir):
    """Test that the FAISSVectorStore initializes correctly."""
    assert faiss_store.index_dir == vector_store_dir
    assert faiss_store.model.model_name_or_path == "all-MiniLM-L6-v2"
    assert faiss_store.indices == {}


def test_create_index(faiss_store):
    """Test index creation."""
    session_id = "test-session"

    # Create an index
    with patch.object(faiss_store, "_save_index"):
        # Mock methods to avoid actually saving to disk
        faiss_store._get_or_create_index(session_id)

    # Check that the index exists
    assert session_id in faiss_store.indices

    # Since we mocked _save_index, we're not actually checking the file
    # as that would have side effects in tests


@patch("app.adapters.vector_faiss.SentenceTransformer")
def test_add_document(mock_transformer, faiss_store):
    """Test adding a document to the index."""
    # Setup mock embedding
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_transformer.return_value = mock_model
    faiss_store.model = mock_model

    session_id = "test-session"

    # Setup for memory storage
    faiss_store.indices[session_id] = MagicMock()
    faiss_store.metadata[session_id] = {}

    # Mock save index to avoid disk operations
    with patch.object(faiss_store, "_save_index"):
        # Add a memory
        memory_id = faiss_store.add_memory(
            session_id=session_id,
            text="This is a test document",
            metadata={"source": "test", "id": "123"},
        )

    # Check that the memory was added to metadata
    assert memory_id in faiss_store.metadata[session_id]
    assert faiss_store.metadata[session_id][memory_id]["text"] == "This is a test document"
    assert "source" in faiss_store.metadata[session_id][memory_id]
    assert faiss_store.metadata[session_id][memory_id]["source"] == "test"


@patch("app.adapters.vector_faiss.SentenceTransformer")
def test_add_documents(mock_transformer, faiss_store):
    """Test adding multiple documents to the index."""
    # Setup mock embedding
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_transformer.return_value = mock_model
    faiss_store.model = mock_model

    session_id = "test-session"

    # Setup for memory storage
    faiss_store.indices[session_id] = MagicMock()
    faiss_store.metadata[session_id] = {}

    # Mock save index to avoid disk operations
    with patch.object(faiss_store, "_save_index"):
        # Add multiple memories
        memory_id1 = faiss_store.add_memory(
            session_id=session_id,
            text="Document 1",
            metadata={"source": "test", "id": "1"},
        )

        memory_id2 = faiss_store.add_memory(
            session_id=session_id,
            text="Document 2",
            metadata={"source": "test", "id": "2"},
        )

    # Check that the memories were added to metadata
    assert memory_id1 in faiss_store.metadata[session_id]
    assert memory_id2 in faiss_store.metadata[session_id]
    assert faiss_store.metadata[session_id][memory_id1]["text"] == "Document 1"
    assert faiss_store.metadata[session_id][memory_id2]["text"] == "Document 2"


@patch("app.adapters.vector_faiss.SentenceTransformer")
def test_similarity_search(mock_transformer, faiss_store):
    """Test similarity search."""
    # Setup mock embedding
    mock_model = MagicMock()
    # For document embeddings
    mock_model.encode.side_effect = [
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),  # For adding docs
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),  # For second doc
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),  # For query
    ]
    mock_transformer.return_value = mock_model
    faiss_store.model = mock_model

    session_id = "test-session"

    # Mock index and search
    mock_index = MagicMock()
    mock_index.ntotal = 2
    # Return doc indices [0, 1] with distances [0.1, 0.2]
    mock_index.search.return_value = (
        np.array([[0.1, 0.2]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
    )
    faiss_store.indices[session_id] = mock_index

    # Set up metadata
    memory_id1 = "test-id-1"
    memory_id2 = "test-id-2"
    faiss_store.metadata[session_id] = {
        memory_id1: {
            "id": memory_id1,
            "text": "Document 1",
            "source": "test",
            "index": 0,
        },
        memory_id2: {
            "id": memory_id2,
            "text": "Document 2",
            "source": "test",
            "index": 1,
        },
    }

    # Perform a search
    results = faiss_store.query_memory(session_id, "Search query", top_k=2)

    # Check results
    assert len(results) == 2
    assert results[0]["text"] == "Document 1"
    assert results[1]["text"] == "Document 2"
    assert results[0]["distance"] == 0.1
    assert results[1]["distance"] == 0.2


def test_similarity_search_nonexistent_session(faiss_store):
    """Test similarity search on a nonexistent session."""
    session_id = "nonexistent-session"

    # For nonexistent session, we should get empty results, not an error
    results = faiss_store.query_memory(session_id, "Query", top_k=1)
    assert results == []


def test_get_all_documents(faiss_store):
    """Test retrieving all documents from a session."""
    session_id = "test-session"

    # Set up metadata
    memory_id1 = "test-id-1"
    memory_id2 = "test-id-2"
    faiss_store.metadata[session_id] = {
        memory_id1: {
            "id": memory_id1,
            "text": "Document 1",
            "source": "test",
            "index": 0,
        },
        memory_id2: {
            "id": memory_id2,
            "text": "Document 2",
            "source": "test",
            "index": 1,
        },
    }

    # Get all memories
    results = faiss_store.get_session_memories(session_id)

    # Check results
    assert len(results) == 2
    # Since dict order is not guaranteed, check by content not position
    texts = [r["text"] for r in results]
    assert "Document 1" in texts
    assert "Document 2" in texts


def test_delete_index(faiss_store):
    """Test deleting an index."""
    session_id = "test-session"

    # Setup mocks
    faiss_store.indices[session_id] = MagicMock()
    faiss_store.metadata[session_id] = {"test-id": {"text": "test", "index": 0}}

    # Mock file operations
    with patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
        # Delete session
        result = faiss_store.delete_session_memories(session_id)

        # Check result
        assert result is True

        # Verify clean up
        assert session_id not in faiss_store.indices
        assert session_id not in faiss_store.metadata
        assert mock_remove.call_count == 2  # Index and metadata files
