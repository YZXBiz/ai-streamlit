"""Tests for chat API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_chat_service, get_current_user
from backend.app.domain.models.chat import ChatSession, Message
from backend.app.domain.models.user import User
from backend.app.main import app


@pytest.mark.asyncio
class TestChatRoutes:
    """Tests for the chat API routes."""

    def setup_method(self):
        """Set up test environment."""
        # Create a test client
        self.client = TestClient(app)

        # Create a mock user
        self.user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
        )

        # Mock the current user dependency
        app.dependency_overrides[get_current_user] = lambda: self.user

    def teardown_method(self):
        """Clean up after test."""
        # Remove dependency overrides
        app.dependency_overrides.clear()

    def test_create_session(self):
        """Test creating a new chat session."""
        # Create mock session
        session = ChatSession(
            id=1,
            user_id=self.user.id,
            name="Test Session",
            data_file_id=1,
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )

        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.create_session.return_value = session
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test create session endpoint
        response = self.client.post(
            "/api/v1/sessions",
            json={
                "name": "Test Session",
                "data_file_id": 1,
            },
        )

        # Verify response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["id"] == 1
        assert response_data["user_id"] == self.user.id
        assert response_data["name"] == "Test Session"
        assert response_data["data_file_id"] == 1
        assert "created_at" in response_data
        assert "updated_at" in response_data

        # Verify service call
        mock_chat_service.create_session.assert_called_once_with(
            user=self.user,
            name="Test Session",
            data_file_id=1,
        )

    def test_get_user_sessions(self):
        """Test getting all sessions for a user."""
        # Create mock sessions
        sessions = [
            ChatSession(
                id=1,
                user_id=self.user.id,
                name="Session 1",
                data_file_id=1,
                created_at="2023-01-01T12:00:00",
                updated_at="2023-01-01T12:00:00",
            ),
            ChatSession(
                id=2,
                user_id=self.user.id,
                name="Session 2",
                data_file_id=2,
                created_at="2023-01-02T12:00:00",
                updated_at="2023-01-02T12:00:00",
            ),
        ]

        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_user_sessions.return_value = sessions
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test get sessions endpoint
        response = self.client.get("/api/v1/sessions")

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        assert response_data[0]["id"] == 1
        assert response_data[0]["name"] == "Session 1"
        assert response_data[1]["id"] == 2
        assert response_data[1]["name"] == "Session 2"

        # Verify service call
        mock_chat_service.get_user_sessions.assert_called_once_with(user=self.user)

    def test_get_session(self):
        """Test getting a specific session."""
        # Create mock session
        session = ChatSession(
            id=1,
            user_id=self.user.id,
            name="Test Session",
            data_file_id=1,
            created_at="2023-01-01T12:00:00",
            updated_at="2023-01-01T12:00:00",
        )

        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_session.return_value = session
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test get session endpoint
        response = self.client.get("/api/v1/sessions/1")

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["id"] == 1
        assert response_data["user_id"] == self.user.id
        assert response_data["name"] == "Test Session"
        assert response_data["data_file_id"] == 1

        # Verify service call
        mock_chat_service.get_session.assert_called_once_with(
            session_id=1,
            user=self.user,
        )

    def test_get_session_not_found(self):
        """Test getting a non-existent session."""
        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_session.side_effect = Exception("Session not found")
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test get session endpoint with non-existent ID
        response = self.client.get("/api/v1/sessions/999")

        # Verify response
        assert response.status_code == 500
        assert "Session not found" in response.text

        # Verify service call
        mock_chat_service.get_session.assert_called_once_with(
            session_id=999,
            user=self.user,
        )

    def test_delete_session(self):
        """Test deleting a session."""
        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.delete_session.return_value = None
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test delete session endpoint
        response = self.client.delete("/api/v1/sessions/1")

        # Verify response
        assert response.status_code == 204

        # Verify service call
        mock_chat_service.delete_session.assert_called_once_with(
            session_id=1,
            user=self.user,
        )

    def test_send_message(self):
        """Test sending a message and getting a response."""
        # Create mock session and message
        message = Message(
            id=1,
            session_id=1,
            content="Response to the query",
            role="assistant",
            created_at="2023-01-01T12:01:00",
        )

        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.send_message.return_value = message
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test send message endpoint
        response = self.client.post(
            "/api/v1/sessions/1/messages",
            json={
                "content": "What is the average sales?",
            },
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["id"] == 1
        assert response_data["session_id"] == 1
        assert response_data["content"] == "Response to the query"
        assert response_data["role"] == "assistant"
        assert "created_at" in response_data

        # Verify service call
        mock_chat_service.send_message.assert_called_once_with(
            session_id=1,
            user=self.user,
            content="What is the average sales?",
        )

    def test_get_messages(self):
        """Test getting all messages for a session."""
        # Create mock messages
        messages = [
            Message(
                id=1,
                session_id=1,
                content="What is the average sales?",
                role="user",
                created_at="2023-01-01T12:00:00",
            ),
            Message(
                id=2,
                session_id=1,
                content="The average sales is $1,000.",
                role="assistant",
                created_at="2023-01-01T12:01:00",
            ),
        ]

        # Create mock chat service
        mock_chat_service = AsyncMock()
        mock_chat_service.get_messages.return_value = messages
        app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        # Test get messages endpoint
        response = self.client.get("/api/v1/sessions/1/messages")

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        assert response_data[0]["id"] == 1
        assert response_data[0]["role"] == "user"
        assert response_data[0]["content"] == "What is the average sales?"
        assert response_data[1]["id"] == 2
        assert response_data[1]["role"] == "assistant"
        assert response_data[1]["content"] == "The average sales is $1,000."

        # Verify service call
        mock_chat_service.get_messages.assert_called_once_with(
            session_id=1,
            user=self.user,
        )
