"""Clustering analysis package for data processing and model building.

This package provides a structured framework for running clustering analysis
pipelines using Dagster for workflow orchestration.
"""

__version__ = "0.1.0"

# Basic utility functions that don't have complex dependencies
from clustering.utils import (
    ensure_directory,
    get_cpu_usage,
    get_memory_usage,
    get_project_root,
    get_system_info,
    profile,
    timer,
)

__all__ = [
    "__version__",
    "ensure_directory",
    "get_cpu_usage",
    "get_memory_usage",
    "get_project_root",
    "get_system_info",
    "profile", 
    "timer",
]

# More complex imports are done within the modules where they're needed
