"""Dagster assets for the clustering pipeline."""

# Preprocessing assets
# Clustering assets - use direct imports
from clustering.dagster.assets.clustering import (
    # No longer importing assign_clusters directly as it doesn't exist
    # assign_clusters,
    # Import the prefixed versions instead
    external_assign_clusters,
    external_calculate_cluster_metrics,
    external_dimensionality_reduced_features,
    external_fe_raw_data,
    external_feature_metadata,
    external_filtered_features,
    external_generate_cluster_visualizations,
    external_imputed_features,
    external_normalized_data,
    external_optimal_cluster_counts,
    external_outlier_removed_features,
    external_save_cluster_assignments,
    external_save_clustering_models,
    external_train_clustering_models,
    internal_assign_clusters,
    internal_calculate_cluster_metrics,
    internal_dimensionality_reduced_features,
    internal_fe_raw_data,
    internal_feature_metadata,
    internal_filtered_features,
    internal_generate_cluster_visualizations,
    internal_imputed_features, 
    internal_normalized_data,
    internal_optimal_cluster_counts,
    internal_outlier_removed_features,
    internal_save_cluster_assignments,
    internal_save_clustering_models,
    internal_train_clustering_models,
)

# Preprocessing assets
from clustering.dagster.assets.preprocessing.internal import (
    internal_normalized_sales_data,
    internal_output_sales_table,
    internal_product_category_mapping,
    internal_raw_sales_data,
    internal_sales_by_category,
    internal_sales_with_categories,
)

# Import preprocessing external assets if needed
from clustering.dagster.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)

__all__ = [
    # Preprocessing - internal
    "internal_raw_sales_data",
    "internal_product_category_mapping",
    "internal_sales_with_categories",
    "internal_sales_by_category",
    "internal_output_sales_table",
    "internal_normalized_sales_data",
    # Preprocessing - external
    "external_features_data",
    "preprocessed_external_data",
    # Clustering - Internal ML assets  
    "internal_fe_raw_data",
    "internal_filtered_features",
    "internal_imputed_features",
    "internal_normalized_data",
    "internal_outlier_removed_features",
    "internal_dimensionality_reduced_features",
    "internal_feature_metadata",
    "internal_optimal_cluster_counts",
    "internal_train_clustering_models",
    "internal_save_clustering_models",
    "internal_assign_clusters",
    "internal_save_cluster_assignments",
    "internal_calculate_cluster_metrics",
    "internal_generate_cluster_visualizations",
    # Clustering - External ML assets
    "external_fe_raw_data",
    "external_filtered_features",
    "external_imputed_features",
    "external_normalized_data",
    "external_outlier_removed_features",
    "external_dimensionality_reduced_features",
    "external_feature_metadata",
    "external_optimal_cluster_counts",
    "external_train_clustering_models",
    "external_save_clustering_models",
    "external_assign_clusters",
    "external_save_cluster_assignments",
    "external_calculate_cluster_metrics",
    "external_generate_cluster_visualizations",
]
