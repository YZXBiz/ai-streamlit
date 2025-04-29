"""Message domain model with additional role definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Enum for message role types (compatible with LLM message formats)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @classmethod
    def from_sender(cls, sender_type):
        """Convert MessageSender to MessageRole."""
        # This is a helper method to convert between different enum formats
        mapping = {"USER": cls.USER, "ASSISTANT": cls.ASSISTANT, "SYSTEM": cls.SYSTEM}
        return mapping.get(sender_type.name, cls.USER)

    @property
    def is_user(self) -> bool:
        """Check if the role is USER."""
        return self == self.USER

    @property
    def is_assistant(self) -> bool:
        """Check if the role is ASSISTANT."""
        return self == self.ASSISTANT

    @property
    def is_system(self) -> bool:
        """Check if the role is SYSTEM."""
        return self == self.SYSTEM


@dataclass
class Message:
    """Message domain model representing a single message with role formatting."""

    id: int | None = None
    session_id: int | None = None
    role: MessageRole = MessageRole.USER
    content: str = ""
    created_at: datetime | str = field(default_factory=datetime.now)

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role.is_user

    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message."""
        return self.role.is_assistant

    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.role.is_system
