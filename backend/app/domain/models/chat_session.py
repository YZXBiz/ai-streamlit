"""ChatSession domain model for storing chat interaction data."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .datafile import DataFile


class MessageSender(Enum):
    """Enum for message sender types."""

    USER = auto()
    ASSISTANT = auto()


@dataclass
class Message:
    """Message domain model representing a single message in a chat session."""

    id: int | None = None
    session_id: int | None = None
    sender: MessageSender = MessageSender.USER
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ChatSession:
    """ChatSession domain model representing a chat session with a user."""

    id: int | None = None
    user_id: int | None = None
    name: str = ""
    description: str = ""
    data_file_id: int | None = None
    data_file: DataFile | None = None
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)

    def add_message(self, content: str, sender: MessageSender) -> Message:
        """Add a new message to the session."""
        message = Message(
            session_id=self.id,
            sender=sender,
            content=content,
        )
        self.messages.append(message)
        self.last_active_at = datetime.now()
        return message

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the session."""
        return self.add_message(content, MessageSender.USER)

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message to the session."""
        return self.add_message(content, MessageSender.ASSISTANT)
