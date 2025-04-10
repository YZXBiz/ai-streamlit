"""External data clustering ML assets for the Dagster pipeline."""

from src.clustering.dagster.assets.clustering.external_ml.feature_engineering import (
    fe_raw_data,
    filtered_features,
    imputed_features,
    normalized_data,
    outlier_removed_features,
    dimensionality_reduced_features,
    feature_metadata,
)

from src.clustering.dagster.assets.clustering.external_ml.model_training import (
    optimal_cluster_counts,
    train_clustering_models,
    save_clustering_models,
    assign_clusters,
    save_cluster_assignments,
    calculate_cluster_metrics,
    generate_cluster_visualizations,
)

external_ml_feature_engineering_assets = [
    fe_raw_data,
    filtered_features,
    imputed_features,
    normalized_data,
    outlier_removed_features,
    dimensionality_reduced_features,
    feature_metadata,
]

external_ml_model_training_assets = [
    optimal_cluster_counts,
    train_clustering_models,
    save_clustering_models,
    assign_clusters,
    save_cluster_assignments,
    calculate_cluster_metrics,
    generate_cluster_visualizations,
]

external_ml_assets = external_ml_feature_engineering_assets + external_ml_model_training_assets
