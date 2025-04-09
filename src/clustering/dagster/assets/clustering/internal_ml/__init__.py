"""Internal ML assets for clustering pipeline.

This module exposes Dagster assets for the internal ML pipeline, including feature
engineering, model training, and inference.
"""

# Export the feature engineering assets
from .feature_engineering import (
    dimensionality_reduced_features,
    fe_raw_data,
    feature_metadata,
    filtered_features,
    imputed_features,
    normalized_data,
    outlier_removed_features,
)

# Export the model inference assets
from .model_inference import generate_clusters, internal_clustering_output, load_production_model

# Export the model training assets
from .model_training import train_clustering_model

__all__ = [
    # Feature engineering
    "fe_raw_data",
    "filtered_features",
    "imputed_features",
    "normalized_data",
    "outlier_removed_features",
    "dimensionality_reduced_features",
    "feature_metadata",
    # Model training
    "train_clustering_model",
    # Model inference
    "load_production_model",
    "generate_clusters",
    "internal_clustering_output",
]
