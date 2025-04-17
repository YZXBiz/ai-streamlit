"""Tests for clustering.utils.common module."""

import tempfile
import time
from pathlib import Path

from clustering.utils.common import ensure_directory, get_project_root, timer


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_create_directory(self) -> None:
        """Test that ensure_directory creates a directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "test_dir"
            result = ensure_directory(new_dir)

            assert new_dir.exists()
            assert new_dir.is_dir()
            assert result == new_dir

    def test_directory_already_exists(self) -> None:
        """Test that ensure_directory handles existing directories correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)
            result = ensure_directory(existing_dir)

            assert existing_dir.exists()
            assert existing_dir.is_dir()
            assert result == existing_dir

    def test_nested_directory(self) -> None:
        """Test that ensure_directory creates nested directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "level3"
            result = ensure_directory(nested_dir)

            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert result == nested_dir


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path_object(self) -> None:
        """Test that get_project_root returns a Path object."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_directory_exists(self) -> None:
        """Test that the returned directory exists."""
        result = get_project_root()
        assert result.exists()
        assert result.is_dir()

    def test_is_actual_project_root(self) -> None:
        """Test that the returned path contains expected project files."""
        root = get_project_root()
        # Check for the existence of common project root indicators
        assert any(
            (root / item).exists()
            for item in ["pyproject.toml", "setup.py", "Makefile", "README.md"]
        )


class TestTimer:
    """Tests for timer decorator."""

    def test_timer_returns_result(self) -> None:
        """Test that timer decorator returns the original function result."""

        @timer
        def example_function() -> str:
            return "expected result"

        result = example_function()
        assert result == "expected result"

    def test_timer_preserves_function_metadata(self) -> None:
        """Test that timer decorator preserves function metadata."""

        @timer
        def example_function() -> None:
            """Example docstring."""
            pass

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Example docstring."

    def test_timer_measures_time(self, capsys) -> None:
        """Test that timer decorator measures execution time."""

        @timer
        def slow_function() -> None:
            time.sleep(0.1)  # Short sleep to ensure measurable time

        slow_function()
        captured = capsys.readouterr()

        assert "slow_function took" in captured.out
        assert "seconds to complete" in captured.out
        # Time should be approximately 0.1 seconds (allowing for execution overhead)
        time_output = captured.out.split("took ")[1].split(" seconds")[0]
        measured_time = float(time_output)
        assert 0.05 < measured_time < 0.5
