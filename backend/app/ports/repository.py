"""Repository interfaces for database operations."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.app.domain.models.chat_session import ChatSession, Message
from backend.app.domain.models.datafile import DataFile
from backend.app.domain.models.user import User

# Generic type variable for repository operations
T = TypeVar("T")


class Repository(Generic[T], ABC):
    """Generic repository interface for basic CRUD operations."""

    @abstractmethod
    async def get(self, entity_id: int) -> T | None:
        """Get an entity by ID."""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: int) -> bool:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    async def list_items(self, skip: int = 0, limit: int = 100) -> list[T]:
        """List entities with pagination."""
        pass

    @abstractmethod
    async def get_all(self) -> list[T]:
        """Get all entities."""
        pass


class UserRepository(Repository[User]):
    """Repository interface for User entities."""

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None:
        """Get a user by username."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Get a user by email."""
        pass


class DataFileRepository(Repository[DataFile]):
    """Repository interface for DataFile entities."""

    @abstractmethod
    async def get_by_user(self, user_id: int) -> list[DataFile]:
        """Get all files for a user."""
        pass


class ChatSessionRepository(Repository[ChatSession]):
    """Repository interface for ChatSession entities."""

    @abstractmethod
    async def get_by_user(self, user_id: int) -> list[ChatSession]:
        """Get all chat sessions for a user."""
        pass

    @abstractmethod
    async def get_session_with_file(self, session_id: int) -> ChatSession | None:
        """Get a chat session with its associated file."""
        pass

    @abstractmethod
    async def add_message(self, message: Message) -> Message:
        """Add a message to a chat session."""
        pass

    @abstractmethod
    async def get_messages(self, session_id: int, skip: int = 0, limit: int = 100) -> list[Message]:
        """Get messages for a chat session with pagination."""
        pass
