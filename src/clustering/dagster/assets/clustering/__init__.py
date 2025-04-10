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
    cluster_assignments,
    cluster_metrics,
    cluster_visualizations,
    generate_cluster_visualizations,
    optimal_cluster_counts,
    persisted_cluster_assignments,
    save_cluster_assignments,
    save_clustering_models,
    saved_clustering_models,
    train_clustering_models,
    trained_clustering_models,
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
    "trained_clustering_models",
    "saved_clustering_models",
    # Cluster assignment
    "cluster_assignments",
    "persisted_cluster_assignments",
    # Evaluation
    "cluster_metrics",
    "cluster_visualizations",
]
