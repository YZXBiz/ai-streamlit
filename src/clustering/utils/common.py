"""Common utilities for the clustering pipeline."""

import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")


def ensure_directory(path: str | Path) -> Path:
    """Ensure that a directory exists.

    Args:
        path: Path to directory

    Returns:
        Path object for the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """Get the root directory of the project.

    Returns:
        Path to the project root
    """
    # Start at the current file
    current_path = Path(__file__).resolve()

    # Go up until we find the src directory
    while current_path.name != "src" and current_path.parent != current_path:
        current_path = current_path.parent

    # Return the parent of src (project root)
    if current_path.name == "src":
        return current_path.parent

    # Fallback to using the current working directory
    return Path(os.getcwd())


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure execution time of a function.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds to complete")
        return result

    return cast(Callable[..., T], wrapper)
