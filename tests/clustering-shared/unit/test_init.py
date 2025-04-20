"""Test for the top-level namespace package."""

import importlib


def test_namespace_package():
    """Test that clustering is a namespace package."""
    # Import the module to cover the __init__.py code 
    module = importlib.import_module("clustering")
    
    # Verify it's a namespace package
    assert hasattr(module, "__path__")
    assert isinstance(module.__path__, list) 