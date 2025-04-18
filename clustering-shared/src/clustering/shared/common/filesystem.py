"""Filesystem utilities."""

import os
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Ensure that a directory exists.

    Args:
        path: Path to the directory

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    os.makedirs(path_obj, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file's location
    current_file = Path(__file__).resolve()
    potential_root = current_file.parent
    
    # Check parent directories for Makefile
    while potential_root != potential_root.parent:
        if (potential_root / 'Makefile').exists():
            return potential_root
        potential_root = potential_root.parent
    
    # Fallback: Check from current working directory
    current_dir = Path.cwd()
    if (current_dir / 'Makefile').exists():
        return current_dir
    
    # Last resort: fixed path from module
    return Path(__file__).resolve().parent.parent.parent.parent 