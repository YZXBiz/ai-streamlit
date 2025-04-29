"""Tests for the FAISS vector store adapter."""

import os
import tempfile
from typing import Dict, List

import numpy as np
import pytest
from backend.app.adapters.vector_faiss import FAISSVectorStore


@pytest.fixture
def temp_index_path():
    """Create a temporary directory for testing vector storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestFAISSVectorStore:
    """Test the FAISSVectorStore adapter."""
    
    @pytest.mark.adapter
    def test_init(self, temp_index_path):
        """Test initialization of FAISSVectorStore."""
        vs = FAISSVectorStore(temp_index_path)
        assert os.path.exists(temp_index_path)
        assert vs.index_path == temp_index_path
        
    @pytest.mark.adapter
    def test_store_and_search_embedding(self, temp_index_path):
        """Test storing and searching embeddings."""
        vs = FAISSVectorStore(temp_index_path)
        
        # Create a test embedding
        embedding = np.random.rand(384).astype(np.float32)  # Common embedding size
        metadata = {"text": "test content", "session_id": "123"}
        
        # Store the embedding
        embedding_id = vs.store_embedding(
            embedding=embedding,
            metadata=metadata,
            collection="test_collection"
        )
        
        assert embedding_id is not None
        
        # Search for similar embeddings
        results = vs.search_similar(
            query_embedding=embedding,
            collection="test_collection",
            limit=1
        )
        
        assert len(results) == 1
        assert results[0]["id"] == embedding_id
        assert results[0]["metadata"] == metadata
        assert "score" in results[0]
        assert results[0]["score"] > 0.9  # Should be very similar to itself
    
    @pytest.mark.adapter
    def test_delete_embedding(self, temp_index_path):
        """Test deleting an embedding."""
        vs = FAISSVectorStore(temp_index_path)
        
        # Create and store a test embedding
        embedding = np.random.rand(384).astype(np.float32)
        metadata = {"text": "test content", "session_id": "123"}
        
        embedding_id = vs.store_embedding(
            embedding=embedding,
            metadata=metadata,
            collection="test_collection"
        )
        
        # Verify the embedding exists
        results = vs.search_similar(
            query_embedding=embedding,
            collection="test_collection",
            limit=1
        )
        assert len(results) == 1
        
        # Delete the embedding
        vs.delete_embedding(
            embedding_id=embedding_id,
            collection="test_collection"
        )
        
        # Verify the embedding no longer exists
        results = vs.search_similar(
            query_embedding=embedding,
            collection="test_collection",
            limit=1
        )
        
        # No results should be returned or score should be much lower
        if len(results) > 0:
            assert results[0]["score"] < 0.8  # Threshold to consider not a match
    
    @pytest.mark.adapter
    def test_clear_collection(self, temp_index_path):
        """Test clearing a collection."""
        vs = FAISSVectorStore(temp_index_path)
        
        # Create and store multiple test embeddings
        embeddings = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        
        for i, emb in enumerate(embeddings):
            vs.store_embedding(
                embedding=emb,
                metadata={"text": f"test content {i}", "session_id": "123"},
                collection="test_collection"
            )
        
        # Verify there are embeddings
        results = vs.search_similar(
            query_embedding=embeddings[0],
            collection="test_collection",
            limit=3
        )
        assert len(results) == 3
        
        # Clear the collection
        vs.clear_collection(collection="test_collection")
        
        # Verify there are no embeddings
        results = vs.search_similar(
            query_embedding=embeddings[0],
            collection="test_collection",
            limit=3
        )
        assert len(results) == 0 