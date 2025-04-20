"""Core functionality for the clustering pipeline.

This package contains the core domain logic, data models, and algorithms
used throughout the clustering pipeline, independent of specific infrastructure
or framework implementations.
"""

# Import and expose core schemas
from clustering.shared.schemas.schemas import (
    BaseSchema,
    DataFrameType,
    DistributedDataSchema,
    MergedDataSchema,
    NSMappingSchema,
    SalesSchema,
)

__all__ = [
    "DataFrameType",
    "BaseSchema",
    "SalesSchema",
    "NSMappingSchema",
    "MergedDataSchema",
    "DistributedDataSchema",
]
