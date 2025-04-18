"""Shared utilities for the clustering project."""

__version__ = "0.1.0"

# Direct imports for convenience
from clustering.shared.io.readers.pickle_reader import PickleReader
from clustering.shared.io.writers.pickle_writer import PickleWriter

__all__ = ["PickleReader", "PickleWriter"]

