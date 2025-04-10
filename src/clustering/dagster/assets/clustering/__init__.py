"""Dagster assets for clustering pipeline."""

# Import the asset functions directly from their module paths
from clustering.dagster.assets.clustering.internal_ml.feature_engineering import (
    dimensionality_reduced_features,
    fe_raw_data,
    feature_metadata,
    filtered_features,
    imputed_features,
    normalized_data,
    outlier_removed_features,
)
from clustering.dagster.assets.clustering.internal_ml.model_training import (
    assign_clusters,
    calculate_cluster_metrics,
    generate_cluster_visualizations,
    optimal_cluster_counts,
    save_cluster_assignments,
    save_clustering_models,
    train_clustering_models,
)

# Export all public assets
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
    # Cluster assignment
    "assign_clusters",
    "save_cluster_assignments",
    # Evaluation
    "calculate_cluster_metrics",
    "generate_cluster_visualizations",
]
