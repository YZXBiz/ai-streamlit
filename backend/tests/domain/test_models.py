"""Unit tests for domain models."""

from datetime import datetime

import pytest

from backend.app.domain.models.chat_session import ChatSession, Message, MessageSender
from backend.app.domain.models.datafile import DataFile, FileType
from backend.app.domain.models.user import User


class TestUserModel:
    """Tests for the User domain model."""

    def test_user_creation(self):
        """Test creating a user with default values."""
        user = User(username="testuser", email="test@example.com")

        assert user.id is None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password == ""
        assert user.first_name == ""
        assert user.last_name == ""
        assert user.is_active is True
        assert user.is_admin is False
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)

    def test_user_full_constructor(self):
        """Test creating a user with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)

        user = User(
            id=1,
            username="admin",
            email="admin@example.com",
            hashed_password="hashed_password",
            first_name="Admin",
            last_name="User",
            is_active=True,
            is_admin=True,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert user.id == 1
        assert user.username == "admin"
        assert user.email == "admin@example.com"
        assert user.hashed_password == "hashed_password"
        assert user.first_name == "Admin"
        assert user.last_name == "User"
        assert user.is_active is True
        assert user.is_admin is True
        assert user.created_at == created_at
        assert user.updated_at == updated_at

    def test_full_name_method(self):
        """Test the full_name method."""
        # With first and last name
        user1 = User(
            username="user1",
            email="user1@example.com",
            first_name="John",
            last_name="Doe",
        )
        assert user1.full_name() == "John Doe"

        # Without first and last name
        user2 = User(username="user2", email="user2@example.com")
        assert user2.full_name() == "user2"

        # With only first name
        user3 = User(username="user3", email="user3@example.com", first_name="Jane")
        assert user3.full_name() == "user3"


class TestDataFileModel:
    """Tests for the DataFile domain model."""

    def test_datafile_creation(self):
        """Test creating a data file with default values."""
        data_file = DataFile(
            filename="test.csv",
            original_filename="original.csv",
            file_path="/path/to/file",
            file_type=FileType.CSV,
        )

        assert data_file.id is None
        assert data_file.user_id is None
        assert data_file.filename == "test.csv"
        assert data_file.original_filename == "original.csv"
        assert data_file.file_path == "/path/to/file"
        assert data_file.file_size == 0
        assert data_file.file_type == FileType.CSV
        assert data_file.description == ""
        assert isinstance(data_file.created_at, datetime)
        assert isinstance(data_file.updated_at, datetime)

    def test_datafile_full_constructor(self):
        """Test creating a data file with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)

        data_file = DataFile(
            id=1,
            user_id=2,
            filename="data.csv",
            original_filename="user_data.csv",
            file_path="/path/to/data.csv",
            file_size=1024,
            file_type=FileType.CSV,
            description="Test data file",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert data_file.id == 1
        assert data_file.user_id == 2
        assert data_file.filename == "data.csv"
        assert data_file.original_filename == "user_data.csv"
        assert data_file.file_path == "/path/to/data.csv"
        assert data_file.file_size == 1024
        assert data_file.file_type == FileType.CSV
        assert data_file.description == "Test data file"
        assert data_file.created_at == created_at
        assert data_file.updated_at == updated_at

    def test_file_type_enum(self):
        """Test the FileType enum."""
        assert FileType.CSV.name == "CSV"
        assert FileType.EXCEL.name == "EXCEL"
        assert FileType.PARQUET.name == "PARQUET"
        assert FileType.JSON.name == "JSON"


class TestChatSessionModels:
    """Tests for ChatSession and Message domain models."""

    def test_message_sender_enum(self):
        """Test the MessageSender enum."""
        assert MessageSender.USER.name == "USER"
        assert MessageSender.ASSISTANT.name == "ASSISTANT"

    def test_message_creation(self):
        """Test creating a message with default values."""
        message = Message(content="Hello, world!")

        assert message.id is None
        assert message.session_id is None
        assert message.sender == MessageSender.USER
        assert message.content == "Hello, world!"
        assert isinstance(message.created_at, datetime)

    def test_message_full_constructor(self):
        """Test creating a message with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)

        message = Message(
            id=1,
            session_id=2,
            sender=MessageSender.ASSISTANT,
            content="I can help with that.",
            created_at=created_at,
        )

        assert message.id == 1
        assert message.session_id == 2
        assert message.sender == MessageSender.ASSISTANT
        assert message.content == "I can help with that."
        assert message.created_at == created_at

    def test_chat_session_creation(self):
        """Test creating a chat session with default values."""
        session = ChatSession(name="Test Session")

        assert session.id is None
        assert session.user_id is None
        assert session.name == "Test Session"
        assert session.description == ""
        assert session.data_file_id is None
        assert session.data_file is None
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert isinstance(session.last_active_at, datetime)

    def test_chat_session_full_constructor(self):
        """Test creating a chat session with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)
        last_active_at = datetime(2023, 1, 3, 12, 0, 0)

        messages = [
            Message(id=1, content="Hello"),
            Message(id=2, content="Hi there", sender=MessageSender.ASSISTANT),
        ]

        data_file = DataFile(id=3, filename="test.csv")

        session = ChatSession(
            id=1,
            user_id=2,
            name="Data Analysis",
            description="Analyzing sales data",
            data_file_id=3,
            data_file=data_file,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            last_active_at=last_active_at,
        )

        assert session.id == 1
        assert session.user_id == 2
        assert session.name == "Data Analysis"
        assert session.description == "Analyzing sales data"
        assert session.data_file_id == 3
        assert session.data_file is data_file
        assert len(session.messages) == 2
        assert session.messages[0].content == "Hello"
        assert session.messages[1].content == "Hi there"
        assert session.created_at == created_at
        assert session.updated_at == updated_at
        assert session.last_active_at == last_active_at

    def test_add_message_method(self):
        """Test the add_message method."""
        session = ChatSession(id=1)

        message = session.add_message("Hello", MessageSender.USER)

        assert message.session_id == 1
        assert message.sender == MessageSender.USER
        assert message.content == "Hello"
        assert len(session.messages) == 1
        assert session.messages[0] is message

    def test_add_user_message_method(self):
        """Test the add_user_message method."""
        session = ChatSession(id=1)

        message = session.add_user_message("How are you?")

        assert message.session_id == 1
        assert message.sender == MessageSender.USER
        assert message.content == "How are you?"
        assert len(session.messages) == 1

    def test_add_assistant_message_method(self):
        """Test the add_assistant_message method."""
        session = ChatSession(id=1)

        message = session.add_assistant_message("I'm fine, thanks!")

        assert message.session_id == 1
        assert message.sender == MessageSender.ASSISTANT
        assert message.content == "I'm fine, thanks!"
        assert len(session.messages) == 1
