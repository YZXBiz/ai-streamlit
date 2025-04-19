"""Example of mocking Python modules for testing.

This file demonstrates how to mock Python modules using different techniques:
1. Using sys.modules to replace modules before import
2. Using unittest.mock.patch.dict to patch modules during a test
3. Using pytest fixtures to provide mocks
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class ExampleClient:
    """Example client that would typically make external API calls."""

    def __init__(self):
        """Initialize the client with an API connection."""
        print("Making expensive API connection...")
        # In a real scenario, this might connect to an external service

    def get_data(self, query):
        """Get data from the API."""
        # This would make an actual API call in production
        return f"Data for {query}"


class ServiceThatUsesClient:
    """A service that uses the example client."""

    def __init__(self, client=None):
        """Initialize the service with a client."""
        self.client = client or ExampleClient()

    def process_query(self, query):
        """Process a query using the client."""
        data = self.client.get_data(query)
        return f"Processed: {data}"


def test_with_sys_modules_mocking():
    """Test using sys.modules to mock a module.

    This approach mocks the module before it's imported by the code under test.
    """
    # Create a mock for the ExampleClient
    mock_client_class = MagicMock()
    mock_client = MagicMock()
    mock_client.get_data.return_value = "Mock data"
    mock_client_class.return_value = mock_client

    # Store the original module if it exists
    original_module = sys.modules.get("example_module", None)

    # Create a mock module
    mock_module = MagicMock()
    mock_module.ExampleClient = mock_client_class

    # Replace the module in sys.modules
    sys.modules["example_module"] = mock_module

    try:
        # This would normally import the real module
        from example_module import ExampleClient

        # Create a service using the mocked client
        client = ExampleClient()
        service = ServiceThatUsesClient(client)

        # Test the service
        result = service.process_query("test")
        assert result == "Processed: Mock data"
        mock_client.get_data.assert_called_once_with("test")
    finally:
        # Restore the original module if it existed
        if original_module:
            sys.modules["example_module"] = original_module
        else:
            del sys.modules["example_module"]


def test_with_patch_dict():
    """Test using patch.dict to mock a module during a test.

    This approach patches sys.modules just for the duration of a with block.
    """
    # Create a mock for the ExampleClient
    mock_client_class = MagicMock()
    mock_client = MagicMock()
    mock_client.get_data.return_value = "Mock data from patch"
    mock_client_class.return_value = mock_client

    # Create a mock module
    mock_module = MagicMock()
    mock_module.ExampleClient = mock_client_class

    # Patch sys.modules for the duration of the with block
    with patch.dict("sys.modules", {"example_module": mock_module}):
        # This would normally import the real module
        from example_module import ExampleClient

        # Create a service using the mocked client
        client = ExampleClient()
        service = ServiceThatUsesClient(client)

        # Test the service
        result = service.process_query("test_patch")
        assert result == "Processed: Mock data from patch"
        mock_client.get_data.assert_called_once_with("test_patch")


@pytest.fixture
def mock_client():
    """Fixture providing a mock client for testing."""
    mock_client = MagicMock()
    mock_client.get_data.return_value = "Mock data from fixture"
    return mock_client


def test_with_fixture(mock_client):
    """Test using a pytest fixture to provide a mock.

    This approach is more Pythonic and avoids import mocking when possible.

    Args:
        mock_client: Mock client provided by the fixture.
    """
    # Create a service using the mocked client
    service = ServiceThatUsesClient(mock_client)

    # Test the service
    result = service.process_query("test_fixture")
    assert result == "Processed: Mock data from fixture"
    mock_client.get_data.assert_called_once_with("test_fixture")
