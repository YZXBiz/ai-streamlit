"""Tests for the chat service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.app.domain.models.chat_session import ChatSession, Message, MessageSender
from backend.app.domain.models.datafile import DataFile, FileType
from backend.app.domain.models.user import User
from backend.app.services.chat_service import ChatService


@pytest.mark.asyncio
class TestChatService:
    """Tests for the ChatService class."""

    async def test_create_session_success(self):
        """Test successful creation of a chat session."""
        # Create a mock DataFile
        data_file = DataFile(
            id=1,
            user_id=1,
            filename="test.csv",
            original_filename="sales_data.csv",
            file_path="/path/to/test.csv",
            file_type=FileType.CSV,
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock file_repo.get to return the data file
        file_repo.get.return_value = data_file

        # Mock session_repo.create to return a session with ID
        async def mock_create(session):
            session.id = 1
            return session

        session_repo.create.side_effect = mock_create

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test creating a chat session
        session = await chat_service.create_session(
            user_id=1,
            data_file_id=1,
            name="Test Session",
            description="Testing the chat service",
        )

        # Verify results
        assert session.id == 1
        assert session.user_id == 1
        assert session.data_file_id == 1
        assert session.name == "Test Session"
        assert session.description == "Testing the chat service"

        file_repo.get.assert_called_once_with(1)
        session_repo.create.assert_called_once()

    async def test_create_session_no_data_file(self):
        """Test creating a chat session without a data file."""
        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.create to return a session with ID
        async def mock_create(session):
            session.id = 1
            return session

        session_repo.create.side_effect = mock_create

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test creating a chat session without data file
        session = await chat_service.create_session(
            user_id=1,
            name="General Chat",
            description="Chat without data file",
        )

        # Verify results
        assert session.id == 1
        assert session.user_id == 1
        assert session.data_file_id is None
        assert session.name == "General Chat"
        assert session.description == "Chat without data file"

        file_repo.get.assert_not_called()
        session_repo.create.assert_called_once()

    async def test_create_session_data_file_not_found(self):
        """Test creating a chat session with non-existent data file."""
        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock file_repo.get to return None (file not found)
        file_repo.get.return_value = None

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test creating a chat session with non-existent data file
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.create_session(
                user_id=1,
                data_file_id=999,
                name="Test Session",
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Data file not found" in excinfo.value.detail
        file_repo.get.assert_called_once_with(999)
        session_repo.create.assert_not_called()

    async def test_create_session_wrong_user(self):
        """Test creating a chat session with data file belonging to another user."""
        # Create a mock DataFile belonging to user_id=2
        data_file = DataFile(
            id=1,
            user_id=2,  # Different from the requesting user_id=1
            filename="test.csv",
            original_filename="sales_data.csv",
            file_path="/path/to/test.csv",
            file_type=FileType.CSV,
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock file_repo.get to return the data file
        file_repo.get.return_value = data_file

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test creating a chat session with data file belonging to another user
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.create_session(
                user_id=1,  # Different from data_file.user_id=2
                data_file_id=1,
                name="Test Session",
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Data file not found" in excinfo.value.detail
        file_repo.get.assert_called_once_with(1)
        session_repo.create.assert_not_called()

    async def test_create_session_default_name(self):
        """Test creating a chat session with default name based on data file."""
        # Create a mock DataFile
        data_file = DataFile(
            id=1,
            user_id=1,
            filename="test.csv",
            original_filename="sales_data.csv",
            file_path="/path/to/test.csv",
            file_type=FileType.CSV,
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock file_repo.get to return the data file
        file_repo.get.return_value = data_file

        # Mock session_repo.create to return a session with ID
        async def mock_create(session):
            session.id = 1
            return session

        session_repo.create.side_effect = mock_create

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test creating a chat session with default name
        session = await chat_service.create_session(
            user_id=1,
            data_file_id=1,
        )

        # Verify results
        assert session.id == 1
        assert session.user_id == 1
        assert session.data_file_id == 1
        assert session.name == "Chat about sales_data.csv"

        file_repo.get.assert_called_with(1)
        assert file_repo.get.call_count == 2  # Called once for validation, once for name
        session_repo.create.assert_called_once()

    async def test_get_session_success(self):
        """Test getting a chat session successfully."""
        # Create a mock ChatSession
        session = ChatSession(
            id=1,
            user_id=1,
            name="Test Session",
            messages=[],
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_with_messages to return the session
        session_repo.get_with_messages.return_value = session

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test getting a chat session
        result = await chat_service.get_session(1, 1)

        # Verify results
        assert result == session
        session_repo.get_with_messages.assert_called_once_with(1)

    async def test_get_session_not_found(self):
        """Test getting a non-existent chat session."""
        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_with_messages to return None
        session_repo.get_with_messages.return_value = None

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test getting a non-existent chat session
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.get_session(999, 1)

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat session not found" in excinfo.value.detail
        session_repo.get_with_messages.assert_called_once_with(999)

    async def test_get_session_wrong_user(self):
        """Test getting a chat session belonging to another user."""
        # Create a mock ChatSession belonging to user_id=2
        session = ChatSession(
            id=1,
            user_id=2,  # Different from the requesting user_id=1
            name="Test Session",
            messages=[],
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_with_messages to return the session
        session_repo.get_with_messages.return_value = session

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test getting a chat session belonging to another user
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.get_session(1, 1)  # User ID 1 tries to access session of user ID 2

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat session not found" in excinfo.value.detail
        session_repo.get_with_messages.assert_called_once_with(1)

    async def test_get_user_sessions(self):
        """Test getting all chat sessions for a user."""
        # Create mock sessions
        sessions = [
            ChatSession(id=1, user_id=1, name="Session 1"),
            ChatSession(id=2, user_id=1, name="Session 2"),
        ]

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_by_user_id to return the sessions
        session_repo.get_by_user_id.return_value = sessions

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test getting user sessions
        result = await chat_service.get_user_sessions(1)

        # Verify results
        assert result == sessions
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2
        session_repo.get_by_user_id.assert_called_once_with(1)

    async def test_delete_session_success(self):
        """Test deleting a chat session successfully."""
        # Create a mock ChatSession
        session = ChatSession(
            id=1,
            user_id=1,
            name="Test Session",
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get to return the session
        session_repo.get.return_value = session

        # Mock session_repo.delete to return True
        session_repo.delete.return_value = True

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test deleting a chat session
        result = await chat_service.delete_session(1, 1)

        # Verify results
        assert result is True
        session_repo.get.assert_called_once_with(1)
        session_repo.delete.assert_called_once_with(1)

    async def test_delete_session_not_found(self):
        """Test deleting a non-existent chat session."""
        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get to return None
        session_repo.get.return_value = None

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test deleting a non-existent chat session
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.delete_session(999, 1)

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat session not found" in excinfo.value.detail
        session_repo.get.assert_called_once_with(999)
        session_repo.delete.assert_not_called()

    async def test_delete_session_wrong_user(self):
        """Test deleting a chat session belonging to another user."""
        # Create a mock ChatSession belonging to user_id=2
        session = ChatSession(
            id=1,
            user_id=2,  # Different from the requesting user_id=1
            name="Test Session",
        )

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get to return the session
        session_repo.get.return_value = session

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test deleting a chat session belonging to another user
        with pytest.raises(HTTPException) as excinfo:
            await chat_service.delete_session(
                1, 1
            )  # User ID 1 tries to delete session of user ID 2

        # Verify exception
        assert excinfo.value.status_code == 403
        assert "You don't have permission to delete this session" in excinfo.value.detail
        session_repo.get.assert_called_once_with(1)
        session_repo.delete.assert_not_called()

    async def test_send_message_success(self):
        """Test sending a message and getting a response successfully."""
        # Create a mock ChatSession
        session = ChatSession(
            id=1,
            user_id=1,
            name="Test Session",
            data_file_id=2,
        )

        # Create mock repositories and service
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_with_messages to return the session
        session_repo.get_with_messages.return_value = session

        # Mock _get_query_response to return a response
        response_content = "This is the AI response"

        # Create the chat service with a mock _get_query_response method
        chat_service = ChatService(session_repo, file_repo, data_service)
        chat_service._get_query_response = AsyncMock(return_value=response_content)

        # Mock session_repo.add_message to simulate adding messages
        async def mock_add_message(message):
            if message.sender == MessageSender.USER:
                message.id = 1
            else:
                message.id = 2
            return message

        session_repo.add_message.side_effect = mock_add_message

        # Test sending a message
        user_message, ai_message = await chat_service.send_message(1, 1, "What is the data about?")

        # Verify results
        assert user_message.session_id == 1
        assert user_message.sender == MessageSender.USER
        assert user_message.content == "What is the data about?"
        assert user_message.id == 1

        assert ai_message.session_id == 1
        assert ai_message.sender == MessageSender.ASSISTANT
        assert ai_message.content == response_content
        assert ai_message.id == 2

        session_repo.get_with_messages.assert_called_once_with(1)
        chat_service._get_query_response.assert_called_once_with(session, "What is the data about?")
        assert session_repo.add_message.call_count == 2

    async def test_send_message_error_handling(self):
        """Test error handling when processing a query fails."""
        # Create a mock ChatSession
        session = ChatSession(
            id=1,
            user_id=1,
            name="Test Session",
            data_file_id=2,
        )

        # Create mock repositories and service
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get_with_messages to return the session
        session_repo.get_with_messages.return_value = session

        # Create the chat service with a mock _get_query_response that raises an exception
        chat_service = ChatService(session_repo, file_repo, data_service)
        chat_service._get_query_response = AsyncMock(
            side_effect=Exception("Query processing failed")
        )

        # Mock session_repo.add_message to simulate adding messages
        async def mock_add_message(message):
            if message.sender == MessageSender.USER:
                message.id = 1
            else:
                message.id = 2
            return message

        session_repo.add_message.side_effect = mock_add_message

        # Test sending a message with error in processing
        user_message, ai_message = await chat_service.send_message(1, 1, "What is the data about?")

        # Verify results
        assert user_message.session_id == 1
        assert user_message.sender == MessageSender.USER
        assert user_message.content == "What is the data about?"
        assert user_message.id == 1

        assert ai_message.session_id == 1
        assert ai_message.sender == MessageSender.ASSISTANT
        assert "Failed to process query" in ai_message.content
        assert "Query processing failed" in ai_message.content
        assert ai_message.id == 2

        session_repo.get_with_messages.assert_called_once_with(1)
        chat_service._get_query_response.assert_called_once_with(session, "What is the data about?")
        assert session_repo.add_message.call_count == 2

    async def test_get_messages(self):
        """Test getting messages for a chat session."""
        # Create mock messages
        messages = [
            Message(id=1, session_id=1, sender=MessageSender.USER, content="Hello"),
            Message(id=2, session_id=1, sender=MessageSender.ASSISTANT, content="Hi there"),
        ]

        # Create mock repositories
        session_repo = AsyncMock()
        file_repo = AsyncMock()
        data_service = AsyncMock()

        # Mock session_repo.get to verify session ownership
        session_repo.get.return_value = ChatSession(id=1, user_id=1)

        # Mock session_repo.get_messages to return the messages
        session_repo.get_messages.return_value = messages

        # Create the chat service
        chat_service = ChatService(session_repo, file_repo, data_service)

        # Test getting messages
        result = await chat_service.get_messages(1, 1)

        # Verify results
        assert result == messages
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there"
        session_repo.get.assert_called_once_with(1)
        session_repo.get_messages.assert_called_once_with(1, 0, 100)
