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
    assign_clusters,
    calculate_cluster_metrics,
    generate_cluster_visualizations,
    optimal_cluster_counts,
    save_cluster_assignments,
    save_clustering_models,
    train_clustering_models,
)

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
    "train_clustering_models",
    "save_clustering_models",
    # Cluster prediction
    "assign_clusters",
    "save_cluster_assignments",
    # Model evaluation
    "calculate_cluster_metrics",
    "generate_cluster_visualizations",
]
