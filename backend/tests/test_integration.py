"""Integration tests for backend API.

This module contains integration tests that verify the complete flow of the application,
from API requests to database operations and back.
"""

import io
import json
import os
import sys
from typing import Any, Dict
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response

# Clear any previous mocks from conftest
if "backend.app.main" in sys.modules:
    del sys.modules["backend.app.main"]

# We need to import the real app, not a mock
from backend.app.domain.models.datafile import FileType
from backend.app.main import app as real_app

# Constants
TEST_CSV_CONTENT = """date,region,product,sales
2023-01-01,North,Widget A,1200
2023-01-01,South,Widget B,950
2023-01-02,East,Widget A,1350
2023-01-02,West,Widget B,1100
2023-01-03,North,Widget C,800
"""


@pytest.fixture
def client() -> TestClient:
    """Get a test client for the FastAPI app."""
    return TestClient(real_app)


@pytest.mark.asyncio
async def test_health_endpoint(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_complete_user_flow(client: TestClient):
    """Test the complete user flow from registration to chat."""
    # Step 1: Register a new user
    register_data = {
        "username": "integrationuser",
        "email": "integration@example.com",
        "password": "StrongPass123!",
        "first_name": "Integration",
        "last_name": "Test",
    }
    register_response = client.post("/api/v1/register", json=register_data)
    assert register_response.status_code == 201, f"Register failed: {register_response.text}"
    assert "id" in register_response.json()
    user_id = register_response.json()["id"]

    # Step 2: Login with the new user
    login_data = {
        "username": "integrationuser",
        "password": "StrongPass123!",
    }
    login_response = client.post("/api/v1/login", data=login_data)
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"

    token_data = login_response.json()
    assert "access_token" in token_data
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Step 3: Upload a data file
    file_data = io.BytesIO(TEST_CSV_CONTENT.encode())
    files = {"file": ("test_data.csv", file_data, "text/csv")}
    form_data = {"description": "Integration test data file"}

    upload_response = client.post(
        "/api/v1/files/upload",
        files=files,
        data=form_data,
        headers=headers,
    )

    assert upload_response.status_code == 201, f"File upload failed: {upload_response.text}"
    file_data = upload_response.json()
    assert "id" in file_data
    file_id = file_data["id"]

    # Step 4: Create a chat session
    session_data = {
        "name": "Integration Test Session",
        "description": "Testing the complete flow",
        "data_file_id": file_id,
    }

    session_response = client.post(
        "/api/v1/chat/sessions",
        json=session_data,
        headers=headers,
    )

    assert session_response.status_code == 201, f"Session creation failed: {session_response.text}"
    session = session_response.json()
    assert "id" in session
    session_id = session["id"]

    # Step 5: Send a chat message and get a response
    query_data = {"question": "What is the total sales by region?"}

    query_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/query",
        json=query_data,
        headers=headers,
    )

    assert query_response.status_code == 200, f"Chat query failed: {query_response.text}"
    answer = query_response.json()
    assert "answer" in answer

    # Step 6: Get chat history
    history_response = client.get(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=headers,
    )

    assert history_response.status_code == 200, f"Get history failed: {history_response.text}"
    messages = history_response.json()
    assert isinstance(messages, list)
    assert len(messages) >= 2  # At least user question and assistant response

    # Step 7: Get list of user's sessions
    sessions_response = client.get(
        "/api/v1/chat/sessions",
        headers=headers,
    )

    assert sessions_response.status_code == 200, f"Get sessions failed: {sessions_response.text}"
    sessions = sessions_response.json()
    assert isinstance(sessions, list)
    assert len(sessions) >= 1

    # Step 8: Get list of user's files
    files_response = client.get(
        "/api/v1/files",
        headers=headers,
    )

    assert files_response.status_code == 200, f"Get files failed: {files_response.text}"
    files = files_response.json()
    assert isinstance(files, list)
    assert len(files) >= 1


@pytest.mark.asyncio
async def test_authentication_security(client: TestClient):
    """Test authentication security features."""
    # Attempt to access protected endpoint without token
    response = client.get("/api/v1/files")
    assert response.status_code == 401

    # Attempt with invalid token
    headers = {"Authorization": "Bearer invalidtoken12345"}
    response = client.get("/api/v1/files", headers=headers)
    assert response.status_code == 401

    # Test login with invalid credentials
    login_data = {
        "username": "nonexistentuser",
        "password": "wrongpassword",
    }
    response = client.post("/api/v1/login", data=login_data)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_file_operations(client: TestClient):
    """Test file upload and management operations."""
    # First login to get token
    login_data = {
        "username": "testuser",  # Using test user from conftest.py
        "password": "testpassword",
    }
    login_response = client.post("/api/v1/login", data=login_data)

    if login_response.status_code != 200:
        # Create test user if not exists
        register_data = {
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "testpassword",
            "first_name": "Test",
            "last_name": "User",
        }
        client.post("/api/v1/register", json=register_data)
        login_response = client.post("/api/v1/login", data=login_data)

    assert login_response.status_code == 200

    token_data = login_response.json()
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Test file upload with various file types
    test_files = [
        ("test_data.csv", TEST_CSV_CONTENT, "text/csv"),
        ("test_data.json", json.dumps({"data": [1, 2, 3]}), "application/json"),
    ]

    for filename, content, mime_type in test_files:
        file_data = io.BytesIO(content.encode() if isinstance(content, str) else content)
        files = {"file": (filename, file_data, mime_type)}
        form_data = {"description": f"Test {mime_type} file"}

        response = client.post(
            "/api/v1/files/upload",
            files=files,
            data=form_data,
            headers=headers,
        )

        assert response.status_code == 201
        file_info = response.json()

        # Test file retrieval
        file_id = file_info["id"]
        get_response = client.get(f"/api/v1/files/{file_id}", headers=headers)
        assert get_response.status_code == 200

        # Test file download
        download_response = client.get(f"/api/v1/files/{file_id}/download", headers=headers)
        assert download_response.status_code == 200


@pytest.mark.asyncio
async def test_chat_functionality(client: TestClient):
    """Test chat session creation and interaction."""
    # Login
    login_data = {
        "username": "testuser",
        "password": "testpassword",
    }
    login_response = client.post("/api/v1/login", data=login_data)

    if login_response.status_code != 200:
        # Create test user if not exists
        register_data = {
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "testpassword",
            "first_name": "Test",
            "last_name": "User",
        }
        client.post("/api/v1/register", json=register_data)
        login_response = client.post("/api/v1/login", data=login_data)

    assert login_response.status_code == 200

    token_data = login_response.json()
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Upload a file first
    file_data = io.BytesIO(TEST_CSV_CONTENT.encode())
    files = {"file": ("chat_test.csv", file_data, "text/csv")}
    upload_response = client.post(
        "/api/v1/files/upload",
        files=files,
        data={"description": "Chat test file"},
        headers=headers,
    )
    assert upload_response.status_code == 201
    file_id = upload_response.json()["id"]

    # Create multiple chat sessions
    sessions = []
    for i in range(2):
        session_data = {
            "name": f"Test Session {i}",
            "description": f"Test session {i} description",
            "data_file_id": file_id,
        }

        response = client.post(
            "/api/v1/chat/sessions",
            json=session_data,
            headers=headers,
        )

        assert response.status_code == 201
        sessions.append(response.json())

    # Test interaction with first session
    session_id = sessions[0]["id"]

    # Test multiple queries and follow-ups
    queries = [
        "What is the total sales?",
        "Break it down by region",
        "Which product has the highest sales?",
    ]

    for query in queries:
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/query",
            json={"question": query},
            headers=headers,
        )

        assert response.status_code == 200
        assert "answer" in response.json()

    # Get message history
    history_response = client.get(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=headers,
    )

    assert history_response.status_code == 200
    messages = history_response.json()
    assert len(messages) >= len(queries) * 2  # Each query should have a response


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
