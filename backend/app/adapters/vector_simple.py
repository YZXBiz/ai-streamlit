"""Simple TF-IDF vector store implementation."""

import os
import pickle
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.ports.vectorstore import VectorStore


class SimpleTfidfVectorStore(VectorStore):
    """
    A simple vector store implementation using TF-IDF and cosine similarity.

    This implementation uses scikit-learn's TfidfVectorizer to create embeddings
    and cosine similarity for retrieval. It's not as sophisticated as FAISS or other
    vector stores, but it's simple and works well for small datasets.
    """

    def __init__(self, storage_dir: str):
        """
        Initialize the vector store.

        Args:
            storage_dir: Directory where session data will be stored
        """
        self.storage_dir = storage_dir
        self.sessions = {}  # In-memory cache of sessions

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

    def _get_session_path(self, session_id: int) -> str:
        """Get the file path for a session."""
        return os.path.join(self.storage_dir, f"session_{session_id}.pkl")

    def _load_session(self, session_id: int) -> bool:
        """
        Load a session from disk into memory.

        Returns:
            bool: True if session was loaded, False if not found
        """
        session_path = self._get_session_path(session_id)

        if not os.path.exists(session_path):
            return False

        try:
            with open(session_path, "rb") as f:
                self.sessions[session_id] = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading session {session_id}: {str(e)}")
            return False

    def _save_session(self, session_id: int) -> bool:
        """
        Save a session from memory to disk.

        Returns:
            bool: True if session was saved successfully
        """
        if session_id not in self.sessions:
            return False

        session_path = self._get_session_path(session_id)

        try:
            with open(session_path, "wb") as f:
                pickle.dump(self.sessions[session_id], f)
            return True
        except Exception as e:
            print(f"Error saving session {session_id}: {str(e)}")
            return False

    async def add_memory(
        self, session_id: int, text: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Add a text snippet to the vector store.

        Args:
            session_id: The ID of the chat session
            text: The text to add to the vector store
            metadata: Optional metadata to store with the text

        Returns:
            ID of the stored memory
        """
        # Initialize session if it doesn't exist in memory
        if session_id not in self.sessions:
            # Try to load from disk first
            if not self._load_session(session_id):
                # If not found, create a new session
                self.sessions[session_id] = {
                    "texts": [],
                    "metadata": [],
                }

        # Create metadata dictionary if none provided
        if metadata is None:
            metadata = {}

        # Add text and metadata
        memory_id = str(len(self.sessions[session_id]["texts"]))
        metadata["id"] = memory_id
        metadata["text"] = text

        self.sessions[session_id]["texts"].append(text)
        self.sessions[session_id]["metadata"].append(metadata)

        # Save to disk
        self._save_session(session_id)

        return memory_id

    async def query_memory(
        self, session_id: int, query_text: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for relevant memories.

        Args:
            session_id: The ID of the chat session
            query_text: The text to search for
            top_k: Maximum number of results to return

        Returns:
            List of dictionaries containing text and metadata of relevant memories
        """
        # Check if session exists
        if session_id not in self.sessions:
            if not self._load_session(session_id):
                return []

        # Get texts and metadata for the session
        texts = self.sessions[session_id]["texts"]
        metadata = self.sessions[session_id]["metadata"]

        # If no texts, return empty list
        if not texts:
            return []

        # Add query to the end
        all_texts = texts + [query_text]

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Get query vector (last row)
        query_vector = tfidf_matrix[-1:]

        # Get document vectors (all rows except the last one)
        doc_vectors = tfidf_matrix[:-1]

        # Calculate similarity
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Get top_k indices
        top_k = min(top_k, len(texts))
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Collect results
        results = []
        for idx in top_indices:
            result = metadata[idx].copy()
            result["similarity"] = float(similarities[idx])
            results.append(result)

        return results

    async def delete_session_memories(self, session_id: int) -> bool:
        """
        Delete all memories for a specific session.

        Args:
            session_id: The ID of the chat session

        Returns:
            True if successful
        """
        if session_id not in self.sessions:
            return False

        # Remove from memory
        del self.sessions[session_id]

        # Remove from disk
        session_path = self._get_session_path(session_id)

        try:
            if os.path.exists(session_path):
                os.remove(session_path)
            return True
        except Exception as e:
            print(f"Error deleting session memories: {str(e)}")
            return False

    async def get_session_memories(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all memories for a specific session.

        Args:
            session_id: The ID of the chat session
            limit: Maximum number of memories to return

        Returns:
            List of dictionaries containing text and metadata of memories
        """
        if session_id not in self.sessions:
            if not self._load_session(session_id):
                return []

        # Get all memories for the session
        metadata = self.sessions[session_id]["metadata"]

        # Limit the number of results
        return metadata[-limit:]
