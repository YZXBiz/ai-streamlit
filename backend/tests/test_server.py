"""Tests for the backend server startup.

This module contains simple tests to verify the backend server can start
properly and respond to basic requests.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests


def test_server_health_check():
    """Test that the server can start and respond to health checks."""
    # Skip this test in CI environments where we can't start the server
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping server startup test in CI environment")

    # Start the server in a subprocess
    server_process = subprocess.Popen(
        ["uv", "run", "-m", "backend.app.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # Create a new process group
    )

    try:
        # Wait for server to start (max 10 seconds)
        max_retries = 10
        for i in range(max_retries):
            try:
                # Try to connect to the health check endpoint
                response = requests.get("http://localhost:8000/health")
                if response.status_code == 200:
                    # Server is up and running
                    health_data = response.json()
                    assert health_data["status"] == "ok"
                    assert "version" in health_data
                    assert "api_version" in health_data
                    break
            except requests.ConnectionError:
                # Server not ready yet, wait and retry
                time.sleep(1)

            # If we've tried max_retries times without success, fail the test
            if i == max_retries - 1:
                raise TimeoutError("Server did not start within the expected time")

    finally:
        # Clean up: kill the server process and its children
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait()  # Wait for the process to terminate


def test_api_docs_available():
    """Test that the OpenAPI docs are available.

    This test only runs if the server is already running externally.
    """
    # Skip if the server is not already running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            pytest.skip("Server not running, skipping API docs test")
    except requests.ConnectionError:
        pytest.skip("Server not running, skipping API docs test")

    # Check that the API docs are available
    docs_response = requests.get("http://localhost:8000/docs")
    assert docs_response.status_code == 200
    assert "text/html" in docs_response.headers["Content-Type"]

    # Check that the OpenAPI schema is available
    schema_response = requests.get("http://localhost:8000/openapi.json")
    assert schema_response.status_code == 200
    schema = schema_response.json()

    # Basic schema validation
    assert "openapi" in schema
    assert "paths" in schema
    assert "/api/v1/login" in schema["paths"]
