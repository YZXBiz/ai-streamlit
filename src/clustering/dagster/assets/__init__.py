"""Assets for the clustering pipeline."""

from clustering.dagster.assets.clustering.external import (
    external_cluster_evaluation,
    external_clustering_model,
    external_clustering_output,
    external_clusters,
)
from clustering.dagster.assets.clustering.internal import (
    internal_cluster_evaluation,
    internal_clustering_output,
    internal_clusters,
    normalized_internal_data,
)
from clustering.dagster.assets.merging import merged_clusters, merged_clusters_output
from clustering.dagster.assets.preprocessing.external import external_features_data, preprocessed_external_data
from clustering.dagster.assets.preprocessing.internal import (
    internal_category_data,
    internal_need_state_data,
    internal_sales_data,
    merged_internal_data,
    preprocessed_internal_sales,
    preprocessed_internal_sales_percent,
)

# Add additional assets as they are implemented

__all__ = [
    # Preprocessing assets - Internal
    "internal_sales_data",
    "internal_need_state_data",
    "merged_internal_data",
    "internal_category_data",
    "preprocessed_internal_sales",
    "preprocessed_internal_sales_percent",
    # Preprocessing assets - External
    "external_features_data",
    "preprocessed_external_data",
    # Clustering assets - Internal
    "normalized_internal_data",
    "internal_clusters",
    "internal_cluster_evaluation",
    "internal_clustering_output",
    # Clustering assets - External
    "external_clustering_model",
    "external_clusters",
    "external_cluster_evaluation",
    "external_clustering_output",
    # Merging assets
    "merged_clusters",
    "merged_clusters_output",
]
