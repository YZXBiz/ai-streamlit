"""Dagster assets for clustering pipeline."""

# Import the asset functions directly from their module paths
# Import external assets as needed
from clustering.dagster.assets.clustering.external_ml.feature_engineering import (
    external_dimensionality_reduced_features,
    external_fe_raw_data,
    external_feature_metadata,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_outlier_removed_features,
)
from clustering.dagster.assets.clustering.external_ml.model_training import (
    external_assign_clusters,
    external_calculate_cluster_metrics,
    external_generate_cluster_visualizations,
    external_optimal_cluster_counts,
    external_save_cluster_assignments,
    external_save_clustering_models,
    external_train_clustering_models,
)
from clustering.dagster.assets.clustering.internal_ml.feature_engineering import (
    internal_dimensionality_reduced_features,
    internal_fe_raw_data,
    internal_feature_metadata,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
)
from clustering.dagster.assets.clustering.internal_ml.model_training import (
    internal_assign_clusters,
    internal_calculate_cluster_metrics,
    internal_generate_cluster_visualizations,
    internal_optimal_cluster_counts,
    internal_save_cluster_assignments,
    internal_save_clustering_models,
    internal_train_clustering_models,
)

# Export all public assets
__all__ = [
    # Internal feature engineering
    "internal_fe_raw_data",
    "internal_filtered_features",
    "internal_imputed_features",
    "internal_normalized_data",
    "internal_outlier_removed_features",
    "internal_dimensionality_reduced_features",
    "internal_feature_metadata",
    # Internal model training
    "internal_optimal_cluster_counts",
    "internal_train_clustering_models",
    "internal_save_clustering_models",
    # Internal cluster assignment
    "internal_assign_clusters",
    "internal_save_cluster_assignments",
    # Internal evaluation
    "internal_calculate_cluster_metrics",
    "internal_generate_cluster_visualizations",
    # External feature engineering
    "external_fe_raw_data",
    "external_filtered_features",
    "external_imputed_features",
    "external_normalized_data",
    "external_outlier_removed_features",
    "external_dimensionality_reduced_features",
    "external_feature_metadata",
    # External model training
    "external_optimal_cluster_counts",
    "external_train_clustering_models",
    "external_save_clustering_models",
    # External cluster assignment
    "external_assign_clusters",
    "external_save_cluster_assignments",
    # External evaluation
    "external_calculate_cluster_metrics",
    "external_generate_cluster_visualizations",
]
