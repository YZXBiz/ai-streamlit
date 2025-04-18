"""Clustering assets for the clustering pipeline."""

# Import the asset functions directly from their module paths
# Import external assets as needed
from .external_ml.feature_engineering import (
    external_dimensionality_reduced_features,
    external_fe_raw_data,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_outlier_removed_features,
)
from .external_ml.model_training import (
    external_assign_clusters,
    external_optimal_cluster_counts,
    external_save_cluster_assignments,
    external_save_clustering_models,
    external_train_clustering_models,
)
from .internal_ml.feature_engineering import (
    internal_dimensionality_reduced_features,
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
)
from .internal_ml.model_training import (
    internal_assign_clusters,
    internal_optimal_cluster_counts,
    internal_save_cluster_assignments,
    internal_save_clustering_models,
    internal_train_clustering_models,
)

# Export all public assets
__all__ = [
    # Feature engineering - Internal
    "internal_fe_raw_data",
    "internal_filtered_features",
    "internal_imputed_features",
    "internal_normalized_data",
    "internal_outlier_removed_features",
    "internal_dimensionality_reduced_features",
    # Feature engineering - External
    "external_fe_raw_data",
    "external_filtered_features",
    "external_imputed_features",
    "external_normalized_data",
    "external_outlier_removed_features",
    "external_dimensionality_reduced_features",
    # Model training - Internal
    "internal_optimal_cluster_counts",
    "internal_train_clustering_models",
    "internal_save_clustering_models",
    # Model training - External
    "external_optimal_cluster_counts",
    "external_train_clustering_models",
    "external_save_clustering_models",
    # Cluster assignment - Internal
    "internal_assign_clusters",
    "internal_save_cluster_assignments",
    # Cluster assignment - External
    "external_assign_clusters",
    "external_save_cluster_assignments",
]
