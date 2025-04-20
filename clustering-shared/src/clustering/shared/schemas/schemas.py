"""Data schema definitions for the clustering pipeline.

This module contains Pydantic schemas for validating input and output data
across the pipeline. These schemas ensure data consistency and quality.
"""

# %% IMPORTS
from typing import cast

import pandas as pd
import pandera as pa
import polars as pl
from pandera.typing import Series

# %% TYPES
DataFrameType = pd.DataFrame | pl.DataFrame
SeriesType = pd.Series | pl.Series


# %% Base DataFrameModel for all schemas
class BaseSchema(pa.DataFrameModel):
    """Base class for all schemas with polars support."""

    class Config:
        """Pandera model configuration."""

        coerce = True
        # Set strict to False to allow for extra columns
        strict = False

    @classmethod
    def check(cls, data: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        """Validate input data against the schema.

        Args:
            data: Input data as Pandas or Polars DataFrame

        Returns:
            Validated data, preserving the original type (Pandas or Polars)

        Raises:
            ValueError: If validation fails
        """
        is_polars = isinstance(data, pl.DataFrame)

        # Convert polars to pandas for validation since pandera has better pandas support
        if is_polars:
            pandas_data = data.to_pandas()
        else:
            pandas_data = cast(pd.DataFrame, data)

        # Validate with pandera - let exceptions propagate
        validated_data = cls.validate(pandas_data)

        # Convert back to original type
        if is_polars:
            return pl.from_pandas(validated_data)

        return validated_data


# %% Internal Preprocessing Schemas
class SalesSchema(BaseSchema):
    """Schema for sales input data.

    Contains columns required for sales data processing.
    """

    SKU_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    STORE_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    CAT_DSC: Series[str] = pa.Field(nullable=False)
    TOTAL_SALES: Series[float] = pa.Field(ge=0, nullable=False)

    @pa.check("SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES")
    def check_not_empty(cls, data: pd.DataFrame) -> bool:
        """Check that the DataFrame is not empty."""
        return len(data) > 0


class NSMappingSchema(BaseSchema):
    """Schema for need state input data.

    Defines the structure for need state categorical data.
    """

    PRODUCT_ID: Series[int] = pa.Field(gt=0, nullable=False)
    CATEGORY: Series[str] = pa.Field(nullable=True)  # Allow nullable for more flexibility
    NEED_STATE: Series[str] = pa.Field(nullable=True)  # Allow nullable

    # Make all other fields optional with nullable=True
    CDT: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_1: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_2: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_3: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_4: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_5: Series[str] = pa.Field(nullable=True)
    ATTRIBUTE_6: Series[str] = pa.Field(nullable=True)
    PLANOGRAM_DSC: Series[str] = pa.Field(nullable=True)
    PLANOGRAM_NBR: Series[int] = pa.Field(nullable=True)  # Allow nullable
    NEW_ITEM: Series[bool] = pa.Field(nullable=True)  # Allow nullable
    TO_BE_DROPPED: Series[bool] = pa.Field(nullable=True)  # Allow nullable

    @pa.check("PRODUCT_ID")
    def check_not_empty(cls, data: pd.DataFrame) -> bool:
        """Check that the DataFrame is not empty."""
        return len(data) > 0

    @classmethod
    def relaxed_validate(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Perform a more relaxed validation allowing missing columns.

        Args:
            data: Input DataFrame to validate

        Returns:
            Validated DataFrame with best-effort schema compliance
        """
        # Check only required columns
        required_cols = ["PRODUCT_ID"]
        missing_cols = set(required_cols) - set(data.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert PRODUCT_ID to int if possible
        try:
            data["PRODUCT_ID"] = data["PRODUCT_ID"].astype(int)
        except Exception:
            # If conversion fails, try to ignore errors
            try:
                data["PRODUCT_ID"] = pd.to_numeric(data["PRODUCT_ID"], errors="coerce")
                # Drop rows where the conversion produced NaNs
                data = data.dropna(subset=["PRODUCT_ID"])
                data["PRODUCT_ID"] = data["PRODUCT_ID"].astype(int)
            except Exception as e:
                print(f"Warning: Could not convert PRODUCT_ID column to int: {e}")

        # Convert boolean columns if present
        for bool_col in ["NEW_ITEM", "TO_BE_DROPPED"]:
            if bool_col in data.columns:
                try:
                    data[bool_col] = data[bool_col].map(
                        {"TRUE": True, "FALSE": False, True: True, False: False}
                    )
                except Exception as e:
                    print(f"Warning: Could not convert {bool_col} to boolean: {e}")

        return data


class MergedDataSchema(BaseSchema):
    """Schema for merged input data.

    Combines fields from sales and needs state data.
    """

    SKU_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    STORE_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    CAT_DSC: Series[str] = pa.Field(nullable=False)
    NEED_STATE: Series[str] = pa.Field(nullable=False)
    TOTAL_SALES: Series[float] = pa.Field(ge=0, nullable=False)

    @pa.check("SKU_NBR", "STORE_NBR", "NEED_STATE")
    def check_not_empty(cls, data: pd.DataFrame) -> bool:
        """Check that the DataFrame is not empty."""
        return len(data) > 0


class DistributedDataSchema(BaseSchema):
    """Schema for distributed input data.

    Combines fields from sales and need state data with evenly distributed sales.
    """

    SKU_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    STORE_NBR: Series[int] = pa.Field(gt=0, nullable=False)
    CAT_DSC: Series[str] = pa.Field(nullable=False)
    NEED_STATE: Series[str] = pa.Field(nullable=False)
    TOTAL_SALES: Series[float] = pa.Field(ge=0, nullable=False)

    @pa.check("SKU_NBR", "STORE_NBR", "NEED_STATE")
    def check_not_empty(cls, data: pd.DataFrame) -> bool:
        """Check that the DataFrame is not empty."""
        return len(data) > 0


class InputsSchema(BaseSchema):
    """Base schema for all input data."""

    pass


# Output schemas
class OutputsSchema(BaseSchema):
    """Base schema for all output data.

    Abstract base class for output schemas.
    """

    pass


class TargetsSchema(BaseSchema):
    """Schema for target data.

    Abstract base class for target schemas.
    """

    pass


# Type aliases for use in function signatures
Inputs = DataFrameType
Outputs = DataFrameType
Targets = SeriesType
