"""Dagster assets for the clustering pipeline."""

# Preprocessing assets
# Clustering assets
from clustering.dagster.assets.clustering.external import (
    external_cluster_evaluation,
    external_clustering_model,
    external_clustering_output,
    external_clusters,
)
from clustering.dagster.assets.clustering.internal import (
    internal_cluster_evaluation,
    internal_clustering_model,
    internal_clustering_output,
    internal_clusters,
)

# Merging assets
from clustering.dagster.assets.merging import merged_clusters, merged_clusters_output
from clustering.dagster.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)
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
    # Preprocessing - external
    "external_features_data",
    "preprocessed_external_data",
    # Clustering - internal
    "internal_clustering_model",
    "internal_clusters",
    "internal_cluster_evaluation",
    "internal_clustering_output",
    # Clustering - external
    "external_clustering_model",
    "external_clusters",
    "external_cluster_evaluation",
    "external_clustering_output",
    # Merging
    "merged_clusters",
    "merged_clusters_output",
]
