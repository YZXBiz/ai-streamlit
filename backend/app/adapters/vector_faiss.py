"""
FAISS vector store for efficient similarity search.

This adapter implements a vector store using Facebook AI Similarity Search (FAISS)
for fast and efficient similarity searching of text embeddings.
"""

import os
import pickle
import uuid
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..ports.vectorstore import VectorStore


class FAISSVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore interface.

    This adapter uses FAISS for efficient similarity search and SentenceTransformers
    for generating embeddings from text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAISS vector store.

        Args:
            model_name: Name of the SentenceTransformer model to use for embeddings
        """
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Create a directory for storing FAISS indices
        self.index_dir = os.path.join(settings.DATA_DIR, "vector_indices")
        os.makedirs(self.index_dir, exist_ok=True)

        # Dictionary to store session indices and metadata
        self.indices: dict[int, Any] = {}  # session_id -> (index, metadata_dict)
        self.metadata: dict[int, dict[str, Any]] = {}  # session_id -> {id -> metadata}

        # Load existing indices
        self._load_indices()

    def _get_index_path(self, session_id: int) -> str:
        """Get the path for a session's index file."""
        return os.path.join(self.index_dir, f"session_{session_id}.index")

    def _get_metadata_path(self, session_id: int) -> str:
        """Get the path for a session's metadata file."""
        return os.path.join(self.index_dir, f"session_{session_id}.meta")

    def _load_indices(self) -> None:
        """Load existing indices from disk."""
        for filename in os.listdir(self.index_dir):
            if filename.endswith(".index"):
                try:
                    # Extract session_id from filename
                    session_id = int(filename.split("_")[1].split(".")[0])

                    # Load the index
                    index_path = os.path.join(self.index_dir, filename)
                    index = faiss.read_index(index_path)

                    # Load the metadata
                    metadata_path = self._get_metadata_path(session_id)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "rb") as f:
                            metadata = pickle.load(f)
                    else:
                        metadata = {}

                    # Store in memory
                    self.indices[session_id] = index
                    self.metadata[session_id] = metadata
                except Exception as e:
                    print(f"Error loading index {filename}: {str(e)}")

    def _get_or_create_index(self, session_id: int) -> tuple[Any, dict[str, Any]]:
        """
        Get an existing index or create a new one for a session.

        Returns:
            Tuple of (index, metadata_dict)
        """
        if session_id in self.indices:
            return self.indices[session_id], self.metadata[session_id]

        # Create a new index
        index = faiss.IndexFlatL2(self.embedding_dim)
        metadata = {}

        # Store in memory
        self.indices[session_id] = index
        self.metadata[session_id] = metadata

        return index, metadata

    def _save_index(self, session_id: int) -> None:
        """Save an index and its metadata to disk."""
        if session_id not in self.indices:
            return

        # Save the index
        index_path = self._get_index_path(session_id)
        faiss.write_index(self.indices[session_id], index_path)

        # Save the metadata
        metadata_path = self._get_metadata_path(session_id)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata[session_id], f)

    async def add_memory(
        self, session_id: int, text: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Add a text snippet to the vector store.

        Args:
            session_id: ID of the chat session this memory belongs to
            text: Text content to store
            metadata: Optional metadata to store with the text

        Returns:
            ID of the stored memory
        """
        # Get or create the index
        index, metadata_dict = self._get_or_create_index(session_id)

        # Generate embedding
        embedding = self.model.encode([text])[0]
        embedding_normalized = embedding.reshape(1, -1).astype(np.float32)

        # Add to index
        index.add(embedding_normalized)

        # Generate ID and store metadata
        memory_id = str(uuid.uuid4())
        metadata_entry = {
            "id": memory_id,
            "text": text,
            "index": index.ntotal - 1,  # Index of the added vector
        }

        # Add custom metadata if provided
        if metadata:
            metadata_entry.update(metadata)

        metadata_dict[memory_id] = metadata_entry

        # Save to disk
        self._save_index(session_id)

        return memory_id

    async def query_memory(
        self, session_id: int, query_text: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for relevant memories.

        Args:
            session_id: ID of the chat session to query memories from
            query_text: Text to find similar memories for
            top_k: Number of results to return

        Returns:
            List of dictionaries containing text and metadata of relevant memories
        """
        # Check if index exists
        if session_id not in self.indices:
            return []

        index = self.indices[session_id]
        metadata_dict = self.metadata[session_id]

        # If index is empty, return empty list
        if index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0]
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search the index
        top_k = min(top_k, index.ntotal)  # Can't retrieve more than we have
        distances, indices = index.search(query_embedding, top_k)

        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            # Find the metadata entry with this index
            for _memory_id, entry in metadata_dict.items():
                if entry["index"] == idx:
                    # Create a copy of the entry to avoid modifying the original
                    result = entry.copy()
                    result["distance"] = float(distances[0][i])
                    results.append(result)
                    break

        # Sort by distance (smaller is better)
        results.sort(key=lambda x: x["distance"])

        return results

    async def delete_session_memories(self, session_id: int) -> bool:
        """
        Delete all memories for a specific session.

        Args:
            session_id: ID of the chat session

        Returns:
            True if successful
        """
        if session_id not in self.indices:
            return False

        # Remove from memory
        del self.indices[session_id]
        del self.metadata[session_id]

        # Remove from disk
        index_path = self._get_index_path(session_id)
        metadata_path = self._get_metadata_path(session_id)

        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            return True
        except Exception as e:
            print(f"Error deleting session memories: {str(e)}")
            return False

    async def get_session_memories(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all memories for a specific session.

        Args:
            session_id: ID of the chat session
            limit: Maximum number of memories to return

        Returns:
            List of dictionaries containing text and metadata of memories
        """
        if session_id not in self.metadata:
            return []

        # Get all memories for the session
        memories = list(self.metadata[session_id].values())

        # Sort by index (chronological order)
        memories.sort(key=lambda x: x["index"])

        # Limit the number of results
        return memories[:limit]
