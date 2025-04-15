"""Common utilities for the clustering pipeline."""

import os
import time
import inspect
import psutil
import traceback
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import loguru

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


def get_memory_usage() -> str:
    """Get current memory usage.
    
    Returns:
        String representation of memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return f"{memory_mb:.2f} MB"


def get_cpu_usage() -> str:
    """Get current CPU usage for the process.
    
    Returns:
        String representation of CPU usage percentage
    """
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    return f"{cpu_percent:.1f}%"


def get_system_info() -> dict[str, Any]:
    """Get system information.
    
    Returns:
        Dictionary containing system information
    """
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "memory_percent": f"{psutil.virtual_memory().percent}%",
        "disk_usage": f"{psutil.disk_usage('/').percent}%",
    }


def format_error(e: Exception) -> str:
    """Format an exception with traceback information.
    
    Args:
        e: The exception to format
        
    Returns:
        Formatted error message with traceback
    """
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    return "".join(tb)


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure execution time and memory usage of a function.

    This decorator logs:
    - When the function starts (with argument summary)
    - When the function completes (with execution time and memory usage)
    - If the function fails (with error information)

    Args:
        func: Function to decorate

    Returns:
        Decorated function that logs execution metrics
    """
    logger = loguru.logger

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Format function call for logging
        args_str = ', '.join(repr(arg) for arg in args)
        kwargs_str = ', '.join(f"{k}={repr(v)}" for k, v in kwargs.items())
        call_signature = f"{func.__name__}({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
        
        # Limit the call signature length for log readability
        if len(call_signature) > 100:
            call_signature = f"{call_signature[:97]}..."
        
        # Get memory before execution
        mem_before = get_memory_usage()
        
        # Start timing and log the start
        start_time = time.time()
        logger.debug(f"Starting {call_signature}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            mem_after = get_memory_usage()
            
            # Log successful completion
            logger.info(
                f"{func.__name__} completed in {execution_time:.4f} seconds | "
                f"Memory: {mem_before} → {mem_after}"
            )
            
            return result
        except Exception as e:
            # Calculate execution metrics on failure
            execution_time = time.time() - start_time
            mem_after = get_memory_usage()
            
            # Log the error with context
            logger.error(
                f"{func.__name__} failed after {execution_time:.4f} seconds | "
                f"Memory: {mem_before} → {mem_after} | Error: {str(e)}"
            )
            
            # Re-raise the exception for normal error handling
            raise

    return cast(Callable[..., T], wrapper)


def profile(logger_instance: loguru.Logger = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Create a profiling decorator with an optional custom logger.
    
    This decorator extends the timer decorator with more detailed profiling,
    including CPU usage and detailed memory statistics.
    
    Args:
        logger_instance: Optional custom logger to use
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logger_instance or loguru.logger
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get initial metrics
            start_time = time.time()
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info()
            cpu_before = process.cpu_percent(interval=0.1)
            
            # Format function call details
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            if len(signature) > 80:
                signature = f"{signature[:77]}..."
                
            func_name = f"{func.__module__}.{func.__name__}"
            log.info(f"PROFILE START: {func_name}({signature})")
            
            try:
                # Run the function
                result = func(*args, **kwargs)
                
                # Collect metrics after execution
                execution_time = time.time() - start_time
                memory_after = process.memory_info()
                cpu_after = process.cpu_percent(interval=0.1)
                
                # Calculate deltas
                memory_diff_mb = (memory_after.rss - memory_before.rss) / (1024 * 1024)
                
                # Log results
                log.info(
                    f"PROFILE END: {func_name} | "
                    f"Time: {execution_time:.4f}s | "
                    f"Memory: {memory_before.rss/(1024*1024):.2f}MB → {memory_after.rss/(1024*1024):.2f}MB "
                    f"(Δ {memory_diff_mb:+.2f}MB) | "
                    f"CPU: {cpu_before:.1f}% → {cpu_after:.1f}%"
                )
                
                return result
            except Exception as e:
                # Log failure with profiling info
                execution_time = time.time() - start_time
                log.error(
                    f"PROFILE FAIL: {func_name} failed after {execution_time:.4f}s | "
                    f"Error: {str(e)}"
                )
                raise
                
        return wrapper
    
    return decorator
