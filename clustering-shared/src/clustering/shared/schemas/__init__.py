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

# Import and expose validation utilities
from clustering.shared.schemas.validation import (
    validate_dataframe_schema,
    fix_dataframe_schema,
)

__all__ = [
    # Schema definitions
    "DataFrameType",
    "BaseSchema",
    "SalesSchema",
    "NSMappingSchema",
    "MergedDataSchema",
    "DistributedDataSchema",
    # Validation utilities
    "validate_dataframe_schema",
    "fix_dataframe_schema",
]
