"""Domain models for vector store."""

from typing import Any, Dict, Optional

import numpy as np


class VectorStoreDocument:
    """A document stored in a vector store with its embedding."""

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
    ):
        """Initialize a VectorStoreDocument.

        Args:
            text: The document text
            metadata: Additional metadata for the document
            embedding: Vector embedding of the document
        """
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding

    def __repr__(self) -> str:
        """Return string representation of the document."""
        return f"VectorStoreDocument(text={self.text[:30]}..., metadata={self.metadata})"
