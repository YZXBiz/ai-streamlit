"""Profiling utilities."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import psutil

T = TypeVar("T")


def get_cpu_usage() -> float:
    """Get current CPU usage percentage.

    Returns:
        CPU usage percentage (0-100)
    """
    return psutil.cpu_percent(interval=0.1)


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage.

    Returns:
        Dictionary with memory usage stats
    """
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent,
    }


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def profile(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile function execution.

    Args:
        func: Function to profile

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Get initial resource usage
        start_time = time.time()
        initial_cpu = get_cpu_usage()
        initial_mem = get_memory_usage()

        # Execute function
        result = func(*args, **kwargs)

        # Get final resource usage
        end_time = time.time()
        final_cpu = get_cpu_usage()
        final_mem = get_memory_usage()

        # Calculate and print the profile information
        execution_time = end_time - start_time
        cpu_change = final_cpu - initial_cpu
        mem_change = final_mem["percent"] - initial_mem["percent"]

        print(f"--- Profile for {func.__name__} ---")
        print(f"Time: {execution_time:.4f} seconds")
        print(f"CPU usage: {initial_cpu:.1f}% → {final_cpu:.1f}% (Δ: {cpu_change:.1f}%)")
        print(
            f"Memory usage: {initial_mem['percent']:.1f}% → {final_mem['percent']:.1f}% (Δ: {mem_change:.1f}%)"
        )
        print("----------------------------")

        return result

    return wrapper
