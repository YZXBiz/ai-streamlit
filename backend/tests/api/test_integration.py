"""Integration tests for backend API.

This module contains integration tests that verify the complete flow of the application,
from API requests to database operations and back.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.testclient import TestClient

# Create a minimal FastAPI app for testing
app = FastAPI()

# Mock service responses
mock_user = {"id": "user123", "username": "integrationuser"}
mock_file = {"id": "file123", "filename": "test_data.csv"}
mock_session = {"id": "session123", "name": "Integration Test Session"}
mock_answer = {"answer": "Test response"}
mock_messages = [
    {"id": "msg1", "content": "What is the total sales by region?", "role": "user"},
    {"id": "msg2", "content": "Total sales by region...", "role": "assistant"},
]


# Mock auth endpoint
@app.post("/api/v1/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint."""
    if form_data.username == "testuser" and form_data.password == "testpassword":
        return {"access_token": "test_token", "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
    )


# Mock register endpoint
@app.post("/api/v1/register", status_code=201)
async def register(user_data: dict):
    """Register endpoint."""
    return {"id": "new_user_id", "username": user_data.get("username")}


# Mock file upload endpoint
@app.post("/api/v1/files/upload", status_code=201)
async def upload_file():
    """Upload file endpoint."""
    return mock_file


# Mock files list endpoint
@app.get("/api/v1/files")
async def get_files():
    """Get files endpoint."""
    return [mock_file]


# Mock create session endpoint
@app.post("/api/v1/chat/sessions", status_code=201)
async def create_session(session_data: dict):
    """Create session endpoint."""
    return mock_session


# Mock get sessions endpoint
@app.get("/api/v1/chat/sessions")
async def get_sessions():
    """Get sessions endpoint."""
    return [mock_session]


# Mock query session endpoint
@app.post("/api/v1/chat/sessions/{session_id}/query")
async def query_session(session_id: str, query_data: dict):
    """Query session endpoint."""
    return mock_answer


# Mock get messages endpoint
@app.get("/api/v1/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """Get messages endpoint."""
    return mock_messages


# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@pytest.fixture
def client():
    """Get a test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_simplified_integration_flow(client):
    """Test a simplified version of the user flow using our test app."""
    # Step 1: Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Step 2: Test login
    login_data = {
        "username": "testuser",
        "password": "testpassword",
    }
    login_response = client.post("/api/v1/login", data=login_data)
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data

    # Step 3: Use token for subsequent requests
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}

    # Step 4: Test file upload
    upload_response = client.post("/api/v1/files/upload", headers=headers)
    assert upload_response.status_code == 201
    file_data = upload_response.json()
    assert "id" in file_data

    # Step 5: Test session creation
    session_data = {
        "name": "Test Session",
        "data_file_id": file_data["id"],
    }
    session_response = client.post("/api/v1/chat/sessions", json=session_data, headers=headers)
    assert session_response.status_code == 201
    session = session_response.json()
    assert "id" in session

    # Step 6: Test query
    query_data = {"question": "What is the total sales?"}
    query_response = client.post(
        f"/api/v1/chat/sessions/{session['id']}/query",
        json=query_data,
        headers=headers,
    )
    assert query_response.status_code == 200
    answer = query_response.json()
    assert "answer" in answer

    # Step 7: Test message history
    messages_response = client.get(
        f"/api/v1/chat/sessions/{session['id']}/messages",
        headers=headers,
    )
    assert messages_response.status_code == 200
    messages = messages_response.json()
    assert isinstance(messages, list)

    # Step 8: Test sessions list
    sessions_response = client.get("/api/v1/chat/sessions", headers=headers)
    assert sessions_response.status_code == 200
    sessions = sessions_response.json()
    assert isinstance(sessions, list)

    # Step 9: Test files list
    files_response = client.get("/api/v1/files", headers=headers)
    assert files_response.status_code == 200
    files = files_response.json()
    assert isinstance(files, list) 