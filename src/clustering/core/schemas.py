"""Data schema definitions for the clustering pipeline.

This module contains Pydantic schemas for validating input and output data
across the pipeline. These schemas ensure data consistency and quality.
"""

# %% IMPORTS
from typing import TypeVar

import numpy as np
import pandas as pd
import pandera.polars as pa
import polars as pl

# %% TYPES
TSchema = TypeVar("TSchema", bound="pa.DataFrameModel")

# Raw data type aliases
DataFrameType = pd.DataFrame | pl.DataFrame | np.ndarray
SeriesType = pd.Series | pl.Series | np.ndarray


# %% SCHEMAS
class Schema(pa.DataFrameModel):
    """Base class for all schemas."""

    class Config:
        """Pandera schema configuration."""

        coerce = True
        strict = True

    @classmethod
    def check(cls: type[TSchema], data: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        """Validate input data against the schema.

        Args:
            data: Input data as Pandas or Polars DataFrame

        Returns:
            Validated data, preserving the original type (Pandas or Polars)
        """
        # Validate with Pandera directly - it supports both Pandas and Polars
        validated_data = cls.validate(data)
        return validated_data


# %% Internal Preprocessing Schemas
class SalesSchema(Schema):
    """Schema for sales input data.

    Contains columns required for sales data processing.
    """

    SKU_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    STORE_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    CAT_DSC: str = pa.Field(coerce=True, nullable=False)
    TOTAL_SALES: float = pa.Field(coerce=True, nullable=False, ge=0)

    @pa.dataframe_check
    def has_rows(cls, df: pd.DataFrame | pl.DataFrame) -> bool:
        """Check if the DataFrame has at least one row."""
        return len(df) > 0


class NSMappingSchema(Schema):
    """Schema for need state input data.

    Defines the structure for need state categorical data.
    """

    PRODUCT_ID: int = pa.Field(coerce=True, nullable=False, gt=0)
    CATEGORY: str = pa.Field(coerce=True, nullable=False)
    NEED_STATE: str = pa.Field(coerce=True, nullable=False)
    CDT: str = pa.Field(coerce=True, nullable=False)
    ATTRIBUTE_1: str = pa.Field(coerce=True, nullable=True)
    ATTRIBUTE_2: str = pa.Field(coerce=True, nullable=True)
    ATTRIBUTE_3: str = pa.Field(coerce=True, nullable=True)
    ATTRIBUTE_4: str = pa.Field(coerce=True, nullable=True)
    ATTRIBUTE_5: str = pa.Field(coerce=True, nullable=True)
    ATTRIBUTE_6: str = pa.Field(coerce=True, nullable=True)
    PLANOGRAM_DSC: str = pa.Field(coerce=True, nullable=False)
    PLANOGRAM_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    NEW_ITEM: bool = pa.Field(coerce=True, nullable=False)
    TO_BE_DROPPED: bool = pa.Field(coerce=True, nullable=False)

    @pa.dataframe_check
    def has_rows(cls, df: pd.DataFrame | pl.DataFrame) -> bool:
        """Check if the DataFrame has at least one row."""
        return len(df) > 0


class MergedDataSchema(Schema):
    """Schema for merged input data.

    Combines fields from sales and needs state data.
    """

    SKU_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    STORE_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    CAT_DSC: str = pa.Field(coerce=True, nullable=False)
    NEED_STATE: str = pa.Field(coerce=True, nullable=False)
    TOTAL_SALES: float = pa.Field(coerce=True, nullable=False, ge=0)

    @pa.dataframe_check
    def has_rows(cls, df: pd.DataFrame | pl.DataFrame) -> bool:
        """Check if the DataFrame has at least one row."""
        return len(df) > 0


class DistributedDataSchema(Schema):
    """Schema for distributed input data.

    Combines fields from sales and need state data with evenly distributed sales.
    """

    SKU_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    STORE_NBR: int = pa.Field(coerce=True, nullable=False, gt=0)
    CAT_DSC: str = pa.Field(coerce=True, nullable=False)
    NEED_STATE: str = pa.Field(coerce=True, nullable=False)
    TOTAL_SALES: float = pa.Field(coerce=True, nullable=False, ge=0)

    @pa.dataframe_check
    def has_rows(cls, df: pd.DataFrame | pl.DataFrame) -> bool:
        """Check if the DataFrame has at least one row."""
        return len(df) > 0


class InputsSchema(Schema):
    """Base schema for all input data.

    Abstract base class for input schemas.
    """

    pass


# Output schemas
class OutputsSchema(Schema):
    """Base schema for all output data.

    Abstract base class for output schemas.
    """

    pass


class TargetsSchema(Schema):
    """Schema for target data.

    Abstract base class for target schemas.
    """

    pass


# Type aliases for use in function signatures
Inputs = DataFrameType
Outputs = DataFrameType
Targets = SeriesType
