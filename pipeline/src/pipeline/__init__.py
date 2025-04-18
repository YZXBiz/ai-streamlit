"""Pipeline for clustering data using Dagster."""

from pipeline.definitions import create_definitions, defs, get_definitions

# Don't import defs directly to avoid initialization during import
# which can cause issues with Dagster's repository loading
__all__ = ["create_definitions", "get_definitions", "defs"]

__version__ = "0.1.0"
