"""Internal ML assets for clustering pipeline.

This module exposes Dagster assets for the internal ML pipeline, including feature
engineering and model training.
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

# Export the model training and prediction assets
from .model_training import (
    cluster_assignments,
    cluster_metrics,
    cluster_visualizations,
    optimal_cluster_counts,
    persisted_cluster_assignments,
    save_clustering_models,
    saved_clustering_models,
    train_clustering_models,
    trained_clustering_models,
)

# Retain model usage assets separately
from .model_usage import generate_clusters, internal_clustering_output, load_production_model

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
    "optimal_cluster_counts",
    "trained_clustering_models",
    "saved_clustering_models",
    # Cluster prediction
    "cluster_assignments",
    "persisted_cluster_assignments",
    # Model evaluation
    "cluster_metrics",
    "cluster_visualizations",
    # Model usage
    "load_production_model",
    "generate_clusters",
    "internal_clustering_output",
]
