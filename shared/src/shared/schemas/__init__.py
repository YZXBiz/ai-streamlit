"""Core functionality for the clustering pipeline.

This package contains the core domain logic, data models, and algorithms
used throughout the clustering pipeline, independent of specific infrastructure
or framework implementations.
"""

# Import and expose core schemas
from shared.schemas.schemas import (
    SalesSchema,
    NSMappingSchema,
    MergedDataSchema,
    DistributedDataSchema,
)

__all__ = [
    "SalesSchema",
    "NSMappingSchema",
    "MergedDataSchema",
    "DistributedDataSchema",
]
