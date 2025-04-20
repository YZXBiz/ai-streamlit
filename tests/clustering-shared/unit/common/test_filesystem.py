"""Tests for filesystem utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from clustering.shared.common.filesystem import ensure_directory, get_project_root


def test_ensure_directory():
    """Test ensuring a directory exists."""
    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = os.path.join(tmp_dir, "test_dir")
        
        # Directory doesn't exist yet
        assert not os.path.exists(test_dir)
        
        # Create it
        result = ensure_directory(test_dir)
        
        # Verify it exists and was returned correctly
        assert os.path.exists(test_dir)
        assert isinstance(result, Path)
        assert str(result) == test_dir
        
        # Test again with Path object instead of string
        test_dir2 = os.path.join(tmp_dir, "test_dir2")
        result2 = ensure_directory(Path(test_dir2))
        assert os.path.exists(test_dir2)
        assert isinstance(result2, Path)
        assert str(result2) == test_dir2


@patch("pathlib.Path.exists")
@patch("pathlib.Path.resolve")
def test_get_project_root(mock_resolve, mock_exists):
    """Test getting the project root directory."""
    # Mock behavior to simulate finding a Makefile
    mock_exists.return_value = True
    
    # Mock resolve to return predictable paths
    module_path = Path("/fake/path/clustering-shared/src/clustering/shared/common/errors.py")
    mock_resolve.return_value = module_path
    
    # Test the function
    result = get_project_root()
    
    # We should have called exists at least once with a Makefile path
    assert mock_exists.called
    assert isinstance(result, Path)
    
    # Test the fallback case where Makefile is not found anywhere
    mock_exists.return_value = False
    with patch("pathlib.Path.cwd", return_value=Path("/fake/cwd")):
        result = get_project_root()
        assert isinstance(result, Path) 