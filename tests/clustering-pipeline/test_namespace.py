"""Tests for namespace package initialization."""

import importlib
import sys
from unittest.mock import patch


def test_namespace_initialization():
    """Test that the clustering namespace package is properly initialized."""
    # Remove the module if it's already imported
    if "clustering" in sys.modules:
        del sys.modules["clustering"]

    # Mock pkgutil.extend_path to verify it's called correctly
    with patch("pkgutil.extend_path") as mock_extend_path:
        mock_extend_path.return_value = ["mock_path"]

        # Import the module
        importlib.import_module("clustering")

        # Verify that extend_path was called with correct parameters
        mock_extend_path.assert_called_once()
        args, kwargs = mock_extend_path.call_args
        assert len(args) == 2
        assert args[1] == "clustering"  # __name__ parameter is "clustering"
