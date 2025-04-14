"""Dagster implementation for clustering pipeline."""

from clustering.dagster.definitions import create_definitions, get_definitions, defs

# Don't import defs directly to avoid initialization during import
# which can cause issues with Dagster's repository loading
__all__ = ["create_definitions", "get_definitions", "defs"]
