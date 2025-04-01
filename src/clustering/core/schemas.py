# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd
import polars as pl

# %% TYPES
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")


# %% SCHEMAS
class Schema(pa.DataFrameModel):
    class Config:
        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        """Validate input data against the schema.

        Args:
            data: Input data as Pandas or Polars DataFrame

        Returns:
            Validated data, preserving the original type (Pandas or Polars)
        """
        is_polars = isinstance(data, pl.DataFrame)

        # Convert Polars to Pandas for validation
        if is_polars:
            pandas_data = data.to_pandas()
        else:
            pandas_data = data

        # Validate with Pandera
        validated_pandas = T.cast(papd.DataFrame[TSchema], cls.validate(check_obj=pandas_data))

        # Convert back to Polars if the input was Polars
        if is_polars:
            return pl.from_pandas(validated_pandas)
        return validated_pandas


# %% SCHEMAS
class InputsSalesSchema(Schema):
    SKU_NBR: int = pa.Field(nullable=False, gt=0)
    STORE_NBR: int = pa.Field(nullable=False, gt=0)
    CAT_DSC: str = pa.Field(nullable=False)
    TOTAL_SALES: float = pa.Field(nullable=False, ge=0)


class InputsNSSchema(Schema):
    PRODUCT_ID: int = pa.Field(nullable=False, gt=0)
    CATEGORY: str = pa.Field(nullable=False)
    NEED_STATE: str = pa.Field(nullable=False)
    CDT: str = pa.Field(nullable=False)
    ATTRIBUTE_1: str = pa.Field(nullable=True)
    ATTRIBUTE_2: str = pa.Field(nullable=True)
    ATTRIBUTE_3: str = pa.Field(nullable=True)
    ATTRIBUTE_4: str = pa.Field(nullable=True)
    ATTRIBUTE_5: str = pa.Field(nullable=True)
    ATTRIBUTE_6: str = pa.Field(nullable=True)
    PLANOGRAM_DSC: str = pa.Field(nullable=False)
    PLANOGRAM_NBR: int = pa.Field(nullable=False, gt=0)
    NEW_ITEM: bool = pa.Field(nullable=False)
    TO_BE_DROPPED: bool = pa.Field(nullable=False)


class InputsMergedSchema(Schema):
    SKU_NBR: int = pa.Field(nullable=False, gt=0)
    STORE_NBR: int = pa.Field(nullable=False, gt=0)
    CAT_DSC: str = pa.Field(nullable=False)
    NEED_STATE: str = pa.Field(nullable=False)
    TOTAL_SALES: float = pa.Field(nullable=False, ge=0)


class InputsSchema(Schema):
    pass


Inputs = InputsSchema  # transformed inputs ready for modeling


class OuputsSchema(Schema):
    pass


Outputs = InputsSchema


class TargetsSchema(Schema):
    pass


Targets = TargetsSchema  # transformed targets ready for modeling
