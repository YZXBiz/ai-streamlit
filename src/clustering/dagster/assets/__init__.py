"""Dagster assets for the clustering pipeline."""

# Preprocessing assets
# Clustering assets - use direct imports
from clustering.dagster.assets.clustering import (
    assign_clusters,
    calculate_cluster_metrics,
    dimensionality_reduced_features,
    fe_raw_data,
    feature_metadata,
    filtered_features,
    generate_cluster_visualizations,
    imputed_features,
    normalized_data,
    optimal_cluster_counts,
    outlier_removed_features,
    save_cluster_assignments,
    save_clustering_models,
    train_clustering_models,
)

# Preprocessing assets
from clustering.dagster.assets.preprocessing.internal import (
    normalized_sales_data,
    output_sales_table,
    product_category_mapping,
    raw_sales_data,
    sales_by_category,
    sales_with_categories,
)

__all__ = [
    # Preprocessing - internal
    "raw_sales_data",
    "product_category_mapping",
    "sales_with_categories",
    "sales_by_category",
    "output_sales_table",
    "normalized_sales_data",
    # Clustering - ML assets
    "fe_raw_data",
    "filtered_features",
    "imputed_features",
    "normalized_data",
    "outlier_removed_features",
    "dimensionality_reduced_features",
    "feature_metadata",
    "optimal_cluster_counts",
    "train_clustering_models",
    "save_clustering_models",
    "assign_clusters",
    "save_cluster_assignments",
    "calculate_cluster_metrics",
    "generate_cluster_visualizations",
]
