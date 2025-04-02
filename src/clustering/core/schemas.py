# %% IMPORTS

import typing as T

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as papd
import polars as pl
from pydantic import BaseModel, Field, model_validator

# %% TYPES
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# Type aliases
Inputs = T.Union[pd.DataFrame, pl.DataFrame, np.ndarray]
Targets = T.Union[pd.Series, pl.Series, np.ndarray]
Outputs = T.Union[pd.DataFrame, pl.DataFrame, np.array]


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


# Data contract definitions for clustering
class ClusteringConfig(BaseModel):
    """Configuration for clustering algorithms."""

    algorithm: str = Field("kmeans", description="Clustering algorithm to use")
    n_clusters: int = Field(5, description="Number of clusters for kmeans", ge=2)
    random_state: int = Field(42, description="Random state for reproducibility")
    min_cluster_size: int = Field(5, description="Minimum cluster size for HDBSCAN", ge=3)
    cluster_selection_epsilon: float = Field(0.0, description="Epsilon for HDBSCAN cluster selection", ge=0.0)

    @model_validator(mode="after")
    def validate_algorithm_params(self) -> "ClusteringConfig":
        """Validate that the parameters match the algorithm."""
        if self.algorithm.lower() == "kmeans" and not hasattr(self, "n_clusters"):
            raise ValueError("n_clusters is required for kmeans algorithm")
        if self.algorithm.lower() == "hdbscan" and not hasattr(self, "min_cluster_size"):
            raise ValueError("min_cluster_size is required for HDBSCAN algorithm")
        return self


class ClusterFeature(BaseModel):
    """Feature values for a cluster."""

    name: str
    mean: float
    median: float = None
    min: float = None
    max: float = None


class ClusterOutputSchema(BaseModel):
    """Schema for cluster output data."""

    cluster_id: str
    size: int = Field(..., gt=0, description="Number of points in the cluster")
    features: list[ClusterFeature] = Field([], description="Statistical features of the cluster")
    silhouette_score: float = Field(None, description="Silhouette score for the cluster", ge=-1, le=1)


class ClusteringResult(BaseModel):
    """Overall result of a clustering operation."""

    model_version: str
    algorithm: str
    parameters: dict
    clusters: list[ClusterOutputSchema]
    metadata: dict = Field(default_factory=dict)
    timestamp: str

    @model_validator(mode="after")
    def validate_clusters(self) -> "ClusteringResult":
        """Validate that clusters exist."""
        if not self.clusters:
            raise ValueError("Clustering result must have at least one cluster")
        return self
