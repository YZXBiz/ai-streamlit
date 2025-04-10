"""Internal ML assets for clustering pipeline.

This module exposes Dagster assets for the internal ML pipeline, including feature
engineering and model training.
"""

# Export the feature engineering assets
from .feature_engineering import (
    internal_dimensionality_reduced_features,
    internal_fe_raw_data,
    internal_feature_metadata,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
)

# Export the model training and prediction assets
from .model_training import (
    internal_assign_clusters,
    internal_calculate_cluster_metrics,
    internal_generate_cluster_visualizations,
    internal_optimal_cluster_counts,
    internal_save_cluster_assignments,
    internal_save_clustering_models,
    internal_train_clustering_models,
)

__all__ = [
    # Feature engineering
    "internal_fe_raw_data",
    "internal_filtered_features",
    "internal_imputed_features",
    "internal_normalized_data",
    "internal_outlier_removed_features",
    "internal_dimensionality_reduced_features",
    "internal_feature_metadata",
    # Model training
    "internal_optimal_cluster_counts",
    "internal_train_clustering_models",
    "internal_save_clustering_models",
    # Cluster prediction
    "internal_assign_clusters",
    "internal_save_cluster_assignments",
    # Model evaluation
    "internal_calculate_cluster_metrics",
    "internal_generate_cluster_visualizations",
]
