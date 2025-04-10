"""Dagster assets for the clustering pipeline."""

# Preprocessing assets
# Clustering assets - use direct imports
from clustering.dagster.assets.clustering import (
    assign_clusters,
    cluster_assignments,
    cluster_metrics,
    cluster_visualizations,
    dimensionality_reduced_features,
    fe_raw_data,
    feature_metadata,
    filtered_features,
    generate_clusters,
    imputed_features,
    internal_clustering_output,
    load_production_model,
    normalized_data,
    optimal_cluster_counts,
    outlier_removed_features,
    persisted_cluster_assignments,
    saved_clustering_models,
    trained_clustering_models,
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
    "trained_clustering_models",
    "saved_clustering_models",
    "cluster_assignments",
    "persisted_cluster_assignments", 
    "cluster_metrics",
    "cluster_visualizations",
    "load_production_model",
    "generate_clusters",
    "internal_clustering_output",
]
