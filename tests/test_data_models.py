"""Tests for data models."""

from datetime import datetime, timezone

import pytest
from app.domain.models.chat_session import ChatSession, Message, MessageSender
from app.domain.models.datafile import DataFile, FileType
from app.domain.models.user import User
from app.domain.models.vector_store import VectorStoreDocument


def test_user_model():
    """Test User model initialization and attributes."""
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="hashedpassword123",
        first_name="Test",
        last_name="User",
        is_active=True,
        is_admin=False,
    )

    assert user.id == 1
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.hashed_password == "hashedpassword123"
    assert user.first_name == "Test"
    assert user.last_name == "User"
    assert user.is_active is True
    assert user.is_admin is False
    assert user.full_name == "Test User"


def test_file_type_enum():
    """Test FileType enumeration values."""
    assert FileType.CSV.value == "CSV"
    assert FileType.EXCEL.value == "EXCEL"
    assert FileType.PARQUET.value == "PARQUET"
    assert FileType.JSON.value == "JSON"
    assert FileType.SQL.value == "SQL"


def test_datafile_model():
    """Test DataFile model initialization and attributes."""
    file = DataFile(
        id=1,
        user_id=1,
        filename="test_data.csv",
        original_filename="sales_data.csv",
        file_path="/data/test_data.csv",
        file_size=1024,
        file_type=FileType.CSV,
        description="Test data file",
    )

    assert file.id == 1
    assert file.user_id == 1
    assert file.filename == "test_data.csv"
    assert file.original_filename == "sales_data.csv"
    assert file.file_path == "/data/test_data.csv"
    assert file.file_size == 1024
    assert file.file_type == FileType.CSV
    assert file.description == "Test data file"


def test_message_sender_enum():
    """Test MessageSender enumeration values."""
    assert MessageSender.USER.value == "USER"
    assert MessageSender.ASSISTANT.value == "ASSISTANT"
    assert MessageSender.SYSTEM.value == "SYSTEM"


def test_message_model():
    """Test Message model initialization and attributes."""
    now = datetime.now(timezone.utc)
    message = Message(
        id=1,
        session_id=1,
        content="Hello, how can I help you?",
        sender=MessageSender.ASSISTANT,
        created_at=now,
    )

    assert message.id == 1
    assert message.session_id == 1
    assert message.content == "Hello, how can I help you?"
    assert message.sender == MessageSender.ASSISTANT
    assert message.created_at == now


def test_chat_session_model():
    """Test ChatSession model initialization and attributes."""
    now = datetime.now(timezone.utc)
    session = ChatSession(
        id=1,
        user_id=1,
        name="Data Analysis Session",
        description="Analyzing sales data",
        data_file_id=1,
        created_at=now,
    )

    assert session.id == 1
    assert session.user_id == 1
    assert session.name == "Data Analysis Session"
    assert session.description == "Analyzing sales data"
    assert session.data_file_id == 1
    assert session.created_at == now
    assert session.messages == []


def test_chat_session_add_message():
    """Test adding messages to a ChatSession."""
    session = ChatSession(
        id=1,
        user_id=1,
        name="Data Analysis Session",
        data_file_id=1,
    )

    # Add a user message
    user_message = session.add_message(
        content="Can you analyze this data?",
        sender=MessageSender.USER,
    )

    assert len(session.messages) == 1
    assert user_message.content == "Can you analyze this data?"
    assert user_message.sender == MessageSender.USER

    # Add an assistant message
    assistant_message = session.add_message(
        content="I'll analyze the data for you.",
        sender=MessageSender.ASSISTANT,
    )

    assert len(session.messages) == 2
    assert assistant_message.content == "I'll analyze the data for you."
    assert assistant_message.sender == MessageSender.ASSISTANT


def test_vector_store_document():
    """Test VectorStoreDocument model initialization and attributes."""
    import numpy as np

    embedding = np.array([0.1, 0.2, 0.3, 0.4])
    metadata = {"source": "test", "timestamp": "2023-01-01"}

    doc = VectorStoreDocument(
        text="This is a test document.",
        metadata=metadata,
        embedding=embedding,
    )

    assert doc.text == "This is a test document."
    assert doc.metadata == metadata
    assert np.array_equal(doc.embedding, embedding)
