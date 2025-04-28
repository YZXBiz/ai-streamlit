"""Test the health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    """Create a test client for the app."""
    return TestClient(app)


def test_health_check(client):
    """Test that the health check endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()
    assert "api_version" in response.json()
