"""Interface for vector store operations."""

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """
    Interface for vector store operations.

    A vector store is used to store and retrieve embeddings of text.
    This enables semantic search and memory for the chat application.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def delete_session_memories(self, session_id: int) -> bool:
        """
        Delete all memories for a specific session.

        Args:
            session_id: The ID of the chat session

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_session_memories(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all memories for a specific session.

        Args:
            session_id: The ID of the chat session
            limit: Maximum number of memories to return

        Returns:
            List of dictionaries containing text and metadata of memories
        """
        pass
