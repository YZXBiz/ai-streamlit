"""Tests for the profiling module."""

import time
from unittest.mock import patch, MagicMock
import pytest

from clustering.shared.common.profiling import (
    get_cpu_usage,
    get_memory_usage,
    timer,
    profile,
)


@pytest.fixture
def mock_time():
    """Fixture for mocking time.time()."""
    with patch("time.time") as mock:
        yield mock


@pytest.fixture
def mock_print():
    """Fixture for mocking print function."""
    with patch("builtins.print") as mock:
        yield mock


@pytest.fixture
def mock_cpu_usage():
    """Fixture for mocking get_cpu_usage function."""
    with patch("clustering.shared.common.profiling.get_cpu_usage") as mock:
        yield mock


@pytest.fixture
def mock_memory_usage():
    """Fixture for mocking get_memory_usage function."""
    with patch("clustering.shared.common.profiling.get_memory_usage") as mock:
        yield mock


class TestProfilingUtils:
    """Tests for profiling utility functions."""

    @patch("psutil.cpu_percent")
    def test_get_cpu_usage(self, mock_cpu_percent: MagicMock) -> None:
        """Test that get_cpu_usage returns the correct CPU usage."""
        mock_cpu_percent.return_value = 45.0
        result = get_cpu_usage()
        assert result == 45.0
        mock_cpu_percent.assert_called_once_with(interval=0.1)

    @patch("psutil.virtual_memory")
    def test_get_memory_usage(self, mock_virtual_memory: MagicMock) -> None:
        """Test that get_memory_usage returns a dictionary with correct memory metrics."""
        mock_memory = MagicMock()
        mock_memory.total = 16000000000  # 16GB
        mock_memory.available = 8000000000  # 8GB
        mock_memory.used = 8000000000  # 8GB
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory

        result = get_memory_usage()
        assert isinstance(result, dict)
        assert result["total"] == 16000000000
        assert result["available"] == 8000000000
        assert result["used"] == 8000000000
        assert result["percent"] == 50.0
        mock_virtual_memory.assert_called_once()

    @pytest.mark.parametrize(
        "cpu_percent,expected",
        [
            (0.0, 0.0),
            (25.5, 25.5),
            (100.0, 100.0),
        ],
    )
    def test_get_cpu_usage_values(
        self, cpu_percent: float, expected: float
    ) -> None:
        """Test get_cpu_usage with different CPU percentages."""
        with patch("psutil.cpu_percent", return_value=cpu_percent):
            result = get_cpu_usage()
            assert result == expected


class TestTimerDecorator:
    """Tests for the timer decorator."""

    def test_timer_decorator(self, mock_time: MagicMock, mock_print: MagicMock) -> None:
        """Test that the timer decorator correctly measures and prints execution time."""
        # Mock time.time() to return first 100, then 105.5 (5.5 second difference)
        mock_time.side_effect = [100, 105.5]

        # Create a test function with the timer decorator
        @timer
        def test_function() -> str:
            return "test result"

        # Call the decorated function
        result = test_function()

        # Verify that the function returned the correct result
        assert result == "test result"

        # Verify that time.time() was called twice
        assert mock_time.call_count == 2

        # Verify that print was called with the execution time
        mock_print.assert_called_once_with("test_function executed in 5.5000 seconds")

    @pytest.mark.parametrize(
        "start_time,end_time,expected_output",
        [
            (0, 1.5, "test_function executed in 1.5000 seconds"),
            (100, 100.001, "test_function executed in 0.0010 seconds"),
            (50, 60, "test_function executed in 10.0000 seconds"),
        ],
    )
    def test_timer_different_durations(
        self, 
        mock_time: MagicMock, 
        mock_print: MagicMock,
        start_time: float,
        end_time: float,
        expected_output: str,
    ) -> None:
        """Test timer decorator with different execution durations."""
        mock_time.side_effect = [start_time, end_time]

        @timer
        def test_function() -> str:
            return "test result"

        result = test_function()
        assert result == "test result"
        mock_print.assert_called_once_with(expected_output)


class TestProfileDecorator:
    """Tests for the profile decorator."""

    def test_profile_decorator(
        self,
        mock_print: MagicMock,
        mock_time: MagicMock,
        mock_memory_usage: MagicMock,
        mock_cpu_usage: MagicMock,
    ) -> None:
        """Test that the profile decorator correctly measures and prints resource usage."""
        # Mock time.time() to return first 100, then 102.75 (2.75 second difference)
        mock_time.side_effect = [100, 102.75]

        # Mock CPU usage to return first 10%, then 15% (5% difference)
        mock_cpu_usage.side_effect = [10.0, 15.0]

        # Mock memory usage to return dictionaries with different values
        initial_mem = {"total": 16000000000, "available": 8000000000, "used": 8000000000, "percent": 30.0}
        final_mem = {"total": 16000000000, "available": 7000000000, "used": 9000000000, "percent": 35.0}
        mock_memory_usage.side_effect = [initial_mem, final_mem]

        # Create a test function with the profile decorator
        @profile
        def test_function() -> str:
            return "test result"

        # Call the decorated function
        result = test_function()

        # Verify that the function returned the correct result
        assert result == "test result"

        # Verify that functions were called the correct number of times
        assert mock_time.call_count == 2
        assert mock_cpu_usage.call_count == 2
        assert mock_memory_usage.call_count == 2

        # Verify that print was called correctly with profile information
        expected_calls = [
            "--- Profile for test_function ---",
            "Time: 2.7500 seconds",
            "CPU usage: 10.0% → 15.0% (Δ: 5.0%)",
            "Memory usage: 30.0% → 35.0% (Δ: 5.0%)",
            "----------------------------",
        ]
        
        for i, expected_call in enumerate(expected_calls):
            assert mock_print.call_args_list[i][0][0] == expected_call

    def test_profile_preserves_function_metadata(self) -> None:
        """Test that the profile decorator preserves the metadata of the wrapped function."""
        # Create a test function with the profile decorator
        @profile
        def test_function() -> str:
            """Test docstring."""
            return "test result"

        # Verify that the function name and docstring are preserved
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."

    @pytest.mark.parametrize(
        "initial_cpu,final_cpu,initial_mem_pct,final_mem_pct",
        [
            (0.0, 50.0, 10.0, 20.0),
            (25.0, 25.0, 30.0, 30.0),  # No change
            (90.0, 80.0, 70.0, 60.0),  # Decrease in usage
        ],
    )
    def test_profile_with_different_resource_changes(
        self,
        mock_print: MagicMock,
        mock_time: MagicMock,
        mock_memory_usage: MagicMock,
        mock_cpu_usage: MagicMock,
        initial_cpu: float,
        final_cpu: float,
        initial_mem_pct: float,
        final_mem_pct: float,
    ) -> None:
        """Test profile decorator with different resource usage patterns."""
        # Mock time
        mock_time.side_effect = [100, 105]
        
        # Mock CPU usage
        mock_cpu_usage.side_effect = [initial_cpu, final_cpu]
        
        # Mock memory usage
        initial_mem = {"total": 16000000000, "available": 8000000000, "used": 8000000000, "percent": initial_mem_pct}
        final_mem = {"total": 16000000000, "available": 7000000000, "used": 9000000000, "percent": final_mem_pct}
        mock_memory_usage.side_effect = [initial_mem, final_mem]
        
        # Call decorated function
        @profile
        def test_function() -> str:
            return "result"
            
        result = test_function()
        assert result == "result"
        
        # Calculate deltas for verification
        cpu_change = final_cpu - initial_cpu
        mem_change = final_mem_pct - initial_mem_pct
        
        # Verify output
        expected_calls = [
            "--- Profile for test_function ---",
            "Time: 5.0000 seconds",
            f"CPU usage: {initial_cpu:.1f}% → {final_cpu:.1f}% (Δ: {cpu_change:.1f}%)",
            f"Memory usage: {initial_mem_pct:.1f}% → {final_mem_pct:.1f}% (Δ: {mem_change:.1f}%)",
            "----------------------------",
        ]
        
        for i, expected_call in enumerate(expected_calls):
            assert mock_print.call_args_list[i][0][0] == expected_call 