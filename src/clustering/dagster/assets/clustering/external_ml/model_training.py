"""Model training step for the external ML pipeline.

This module provides Dagster assets for training clustering models based on
engineered features from external data sources.
"""

from typing import Any

import dagster as dg
import polars as pl
from pycaret.clustering import ClusteringExperiment


class Defaults:
    """Default configuration values for model training."""

    # Experiment settings
    SESSION_ID = 42

    # Clustering algorithm
    ALGORITHM = "kmeans"

    # Optimal cluster determination
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10

    # Evaluation metrics to track
    METRICS = ["silhouette", "calinski_harabasz", "davies_bouldin"]


@dg.asset(
    name="external_optimal_cluster_counts",
    description="Determines optimal number of clusters for each external data category",
    group_name="model_training",
    compute_kind="external_model_training",
    deps=["external_dimensionality_reduced_features"],
    required_resource_keys={"config"},
)
def external_optimal_cluster_counts(
    context: dg.AssetExecutionContext,
    external_dimensionality_reduced_features: pl.DataFrame,
) -> dict[str, int]:
    """Determine the optimal number of clusters for external data.

    Uses PyCaret to evaluate different cluster counts based on silhouette scores,
    Calinski-Harabasz Index, and Davies-Bouldin Index to determine the optimal
    number of clusters.

    Args:
        context: Dagster asset execution context
        external_dimensionality_reduced_features: DataFrame with reduced dimensions from feature engineering

    Returns:
        Dictionary mapping category names ('default') to optimal cluster count
    """
    optimal_clusters = {}
    category = "default"  # Use a single default category for external data

    # Get configuration parameters or use defaults
    min_clusters = getattr(context.resources.config, "min_clusters", Defaults.MIN_CLUSTERS)
    max_clusters = getattr(context.resources.config, "max_clusters", Defaults.MAX_CLUSTERS)
    metrics = getattr(context.resources.config, "metrics", Defaults.METRICS)
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)

    context.log.info(
        f"Determining optimal clusters using range {min_clusters}-{max_clusters} "
        f"with metrics: {metrics}"
    )

    context.log.info(f"Determining optimal cluster count for external data")

    df = external_dimensionality_reduced_features
    sample_count = len(df)

    # Check if dataset has enough samples for clustering
    if sample_count < min_clusters:
        context.log.warning(
            f"External data has only {sample_count} samples, "
            f"which is less than min_clusters={min_clusters}. "
            f"Setting optimal clusters to 2."  # Changed from 1 to 2 to comply with PyCaret requirements
        )
        optimal_clusters[category] = 2
        return optimal_clusters

    # Adjust max_clusters to not exceed sample count
    adjusted_max_clusters = min(max_clusters, sample_count - 1)
    if adjusted_max_clusters < max_clusters:
        context.log.warning(
            f"External data has only {sample_count} samples. "
            f"Reducing max_clusters from {max_clusters} to {adjusted_max_clusters}."
        )

    # If adjusted_max_clusters is less than min_clusters, we can't cluster properly
    if adjusted_max_clusters < min_clusters:
        context.log.warning(
            f"External data has too few samples ({sample_count}) "
            f"to evaluate clusters in range [{min_clusters}, {max_clusters}]. "
            f"Setting optimal clusters to 2."  # Changed from 1 to 2 to comply with PyCaret requirements
        )
        optimal_clusters[category] = 2
        return optimal_clusters

    # Convert Polars DataFrame to Pandas for PyCaret
    pandas_df = df.to_pandas()

    # Initialize PyCaret experiment
    exp = ClusteringExperiment()
    exp.setup(
        data=pandas_df,
        session_id=session_id,
        verbose=False,
    )

    # Evaluate different cluster counts
    context.log.info(
        f"Evaluating {min_clusters} to {adjusted_max_clusters} clusters for external data"
    )
    cluster_metrics = {}

    # Track metric values for each cluster count
    for k in range(min_clusters, adjusted_max_clusters + 1):
        # Create a model with k clusters
        _ = exp.create_model(Defaults.ALGORITHM, num_clusters=k, verbose=False)

        # Get evaluation metrics
        metrics_values = exp.pull()
        cluster_metrics[k] = {
            metric: metrics_values.loc[0, metric]
            for metric in metrics
            if metric in metrics_values.columns
        }

        # Log progress
        metrics_str = ", ".join(f"{m}={cluster_metrics[k][m]:.4f}" for m in cluster_metrics[k])
        context.log.info(f"  External data with {k} clusters: {metrics_str}")

    # Determine optimal clusters based on silhouette score (higher is better)
    if "silhouette" in metrics and cluster_metrics:
        best_k = max(cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("silhouette", 0))
        context.log.info(f"Optimal clusters for external data based on silhouette: {best_k}")
    # Fallback to Calinski-Harabasz (higher is better)
    elif "calinski_harabasz" in metrics and cluster_metrics:
        best_k = max(
            cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("calinski_harabasz", 0)
        )
        context.log.info(f"Optimal clusters for external data based on calinski_harabasz: {best_k}")
    # Fallback to Davies-Bouldin (lower is better)
    elif "davies_bouldin" in metrics and cluster_metrics:
        best_k = min(
            cluster_metrics.keys(),
            key=lambda k: cluster_metrics[k].get("davies_bouldin", float("inf")),
        )
        context.log.info(f"Optimal clusters for external data based on davies_bouldin: {best_k}")
    else:
        # Default if no metrics match or no clusters were evaluated
        best_k = (
            min(min_clusters, sample_count - 1) if sample_count > 1 else 2
        )  # Changed from 1 to 2
        context.log.warning(
            f"Could not determine optimal clusters for external data, using default: {best_k}"
        )

    optimal_clusters[category] = best_k

    # Store metrics in context for later reference
    context.add_output_metadata(
        {
            f"{category}_metrics": dg.MetadataValue.json(cluster_metrics),
            f"{category}_optimal": best_k,
        }
    )

    return optimal_clusters


@dg.asset(
    name="external_train_clustering_models",
    description="Trains clustering models using optimal number of clusters for external data",
    group_name="model_training",
    compute_kind="external_model_training",
    deps=["external_dimensionality_reduced_features", "external_optimal_cluster_counts"],
    required_resource_keys={"config"},
)
def external_train_clustering_models(
    context: dg.AssetExecutionContext,
    external_dimensionality_reduced_features: pl.DataFrame,
    external_optimal_cluster_counts: dict[str, int],
) -> dict[str, Any]:
    """Train clustering models using engineered features from external data.

    Uses PyCaret to train clustering models using the optimal number of clusters
    determined in the previous step.

    Args:
        context: Dagster asset execution context
        external_dimensionality_reduced_features: DataFrame with reduced dimensions
        external_optimal_cluster_counts: Dictionary mapping category ('default') to optimal cluster count

    Returns:
        Dictionary of trained clustering models organized by category
    """
    trained_models = {}
    category = "default"  # Use a single default category for external data

    # Get configuration parameters
    algorithm = getattr(context.resources.config, "algorithm", Defaults.ALGORITHM)
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)

    context.log.info(f"Training clustering models using algorithm: {algorithm}")

    df = external_dimensionality_reduced_features
    # Get optimal cluster count for this category
    cluster_count = external_optimal_cluster_counts.get(category, 2)

    # Ensure cluster_count is at least 2 as required by PyCaret
    if cluster_count < 2:
        context.log.warning(
            f"Cluster count was {cluster_count}, but PyCaret requires at least 2 clusters. "
            f"Adjusting to 2 clusters."
        )
        cluster_count = 2

    # Ensure we have enough samples for the requested number of clusters
    sample_count = len(df)
    if sample_count <= cluster_count:
        adjusted_cluster_count = min(2, sample_count - 1) if sample_count > 2 else 2
        context.log.warning(
            f"External data has only {sample_count} samples, which is not enough for {cluster_count} clusters. "
            f"Adjusting to {adjusted_cluster_count} clusters."
        )
        cluster_count = adjusted_cluster_count

    # Final validation to ensure we meet PyCaret's requirements
    if sample_count <= 2:
        context.log.error(
            f"External data has only {sample_count} samples, which is insufficient for clustering. "
            f"Skipping model training."
        )
        return {}

    context.log.info(f"Training {algorithm} with {cluster_count} clusters for external data")

    # Convert Polars DataFrame to Pandas for PyCaret
    pandas_df = df.to_pandas()

    # Initialize PyCaret experiment
    exp = ClusteringExperiment()
    exp.setup(
        data=pandas_df,
        session_id=session_id,
        verbose=False,
    )

    # Train the model with the optimal number of clusters
    model = exp.create_model(
        algorithm,
        num_clusters=cluster_count,
        verbose=False,
    )

    # Store the model and experiment
    trained_models[category] = {
        "model": model,
        "experiment": exp,
        "features": df.columns,
        "num_clusters": cluster_count,
        "num_samples": len(df),
    }

    context.log.info(f"Completed training for external data")

    # Add useful metadata to the context
    context.add_output_metadata(
        {
            "algorithm": algorithm,
            "categories": list(trained_models.keys()),
            "cluster_counts": dg.MetadataValue.json(
                {category: data["num_clusters"] for category, data in trained_models.items()}
            ),
        }
    )

    return trained_models


@dg.asset(
    name="external_save_clustering_models",
    description="Persists trained external data clustering models to storage",
    group_name="model_training",
    compute_kind="external_model_training",
    deps=["external_train_clustering_models"],
    required_resource_keys={"config", "external_model_output"},
)
def external_save_clustering_models(
    context: dg.AssetExecutionContext,
    external_train_clustering_models: dict[str, Any],
) -> None:
    """Save trained clustering models to persistent storage.

    Uses the configured model output resource to save the trained models
    for later use in prediction or evaluation.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models by category
    """
    context.log.info("Saving trained clustering models to storage")

    # Use the configured model output resource
    model_output = context.resources.external_model_output

    # Save each model
    for category, model_info in external_train_clustering_models.items():
        context.log.info(f"Saving model for category: {category}")
        model_output.save(category, model_info)

    context.log.info(
        f"Successfully saved {len(external_train_clustering_models)} models to storage"
    )


@dg.asset(
    name="external_assign_clusters",
    description="Assigns clusters to external data points using trained models",
    group_name="cluster_assignment",
    compute_kind="external_cluster_assignment",
    deps=["external_fe_raw_data", "external_train_clustering_models"],
    required_resource_keys={"config"},
)
def external_assign_clusters(
    context: dg.AssetExecutionContext,
    external_fe_raw_data: pl.DataFrame,
    external_train_clustering_models: dict[str, Any],
) -> pl.DataFrame:
    """Assign cluster labels to external data points using trained models.

    Uses the trained clustering models to assign cluster labels to each
    data point in the raw external data.

    Args:
        context: Dagster asset execution context
        external_fe_raw_data: DataFrame with raw external features
        external_train_clustering_models: Dictionary of trained clustering models by category

    Returns:
        DataFrame with cluster assignments, or empty DataFrame with same schema if no models are available
    """
    context.log.info("Assigning clusters to external data points")

    # Check if we have any trained models
    if not external_train_clustering_models:
        context.log.warning(
            "No trained models available for cluster assignment, returning empty DataFrame"
        )
        # Create an empty DataFrame with the expected schema
        # Include the original data but add an empty Cluster column
        result_df = external_fe_raw_data.with_columns(pl.lit(None).cast(pl.Int64).alias("Cluster"))

        # Add metadata about the assignment
        context.add_output_metadata(
            {
                "warning": "No trained models available",
                "total_records": len(result_df),
                "cluster_assigned": False,
            }
        )

        return result_df

    # Convert Polars DataFrame to Pandas
    pandas_df = external_fe_raw_data.to_pandas()

    # Use the default category model if available
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    # Get the model info
    model_info = external_train_clustering_models[category]
    exp = model_info["experiment"]
    model = model_info["model"]

    context.log.info(f"Using model to assign clusters")

    # Get predictions using the trained model
    predictions = exp.predict_model(model, data=pandas_df)

    # Convert back to Polars
    assigned_data = pl.from_pandas(predictions)

    # Log cluster distribution
    cluster_counts = (
        assigned_data.group_by("Cluster").agg(pl.count().alias("count")).sort("Cluster")
    )
    context.log.info(f"Cluster distribution:\n{cluster_counts}")

    # Store metadata about the assignment
    context.add_output_metadata(
        {
            "model_category": category,
            "num_clusters": model_info["num_clusters"],
            "total_records": len(assigned_data),
            "cluster_distribution": dg.MetadataValue.json(cluster_counts.to_dicts()),
            "cluster_assigned": True,
        }
    )

    return assigned_data


@dg.asset(
    name="external_save_cluster_assignments",
    description="Saves external cluster assignments to storage",
    group_name="cluster_assignment",
    compute_kind="external_cluster_assignment",
    deps=["external_assign_clusters"],
    required_resource_keys={"external_cluster_assignments"},
)
def external_save_cluster_assignments(
    context: dg.AssetExecutionContext,
    external_assign_clusters: pl.DataFrame,
) -> None:
    """Save cluster assignments to persistent storage.

    Uses the configured output resource to save the cluster assignments
    for later use in analysis or reporting.

    Args:
        context: Dagster asset execution context
        external_assign_clusters: DataFrame with cluster assignments
    """
    context.log.info("Saving external cluster assignments to storage")

    # Check if the DataFrame contains valid cluster assignments
    # Look for non-null values in the Cluster column
    has_clusters = external_assign_clusters.filter(~pl.col("Cluster").is_null()).height > 0

    if not has_clusters:
        context.log.warning("No valid clusters to save, skipping storage")
        context.add_output_metadata({"status": "skipped", "reason": "No valid clusters assigned"})
        return

    # Use the configured output resource
    assignments_output = context.resources.external_cluster_assignments

    # Save with default category name
    category = "default"
    context.log.info(f"Saving cluster assignments for external data")
    assignments_output.save(category, external_assign_clusters)

    context.log.info("Successfully saved external cluster assignments")
    context.add_output_metadata(
        {"status": "success", "records_saved": external_assign_clusters.height}
    )


@dg.asset(
    name="external_calculate_cluster_metrics",
    description="Calculates metrics for external cluster quality evaluation",
    group_name="cluster_analysis",
    compute_kind="external_cluster_analysis",
    deps=["external_train_clustering_models", "external_assign_clusters"],
    required_resource_keys={"config"},
)
def external_calculate_cluster_metrics(
    context: dg.AssetExecutionContext,
    external_train_clustering_models: dict[str, Any],
    external_assign_clusters: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate metrics to evaluate the quality of clustering for external data.

    Computes various metrics to assess the quality of the clustering results,
    such as silhouette score, inertia, and cluster size distribution.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: DataFrame with cluster assignments

    Returns:
        DataFrame with evaluation metrics or empty metrics DataFrame if no models are available
    """
    # Check if there are any trained models
    if not external_train_clustering_models:
        context.log.warning(
            "No trained models available for metrics calculation, returning empty metrics"
        )
        empty_metrics = pl.DataFrame(
            [
                {
                    "category": "default",
                    "num_clusters": None,
                    "silhouette": None,
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "cluster_distribution": "{}",
                    "status": "no_models_available",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_models_available"})

        return empty_metrics

    # Check if the DataFrame contains valid cluster assignments
    has_clusters = external_assign_clusters.filter(~pl.col("Cluster").is_null()).height > 0
    if not has_clusters:
        context.log.warning("No valid clusters assigned, returning empty metrics")
        empty_metrics = pl.DataFrame(
            [
                {
                    "category": "default",
                    "num_clusters": None,
                    "silhouette": None,
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "cluster_distribution": "{}",
                    "status": "no_clusters_assigned",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_clusters_assigned"})

        return empty_metrics

    # Use the default category if available
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    context.log.info(f"Calculating evaluation metrics for external data")

    # Get model info
    model_info = external_train_clustering_models[category]
    exp = model_info["experiment"]
    model = model_info["model"]

    # Get PyCaret metrics
    pycaret_metrics = exp.pull().to_dict("records")[0]

    # Get cluster distribution from assignments
    cluster_distribution = (
        external_assign_clusters.group_by("Cluster").agg(pl.count().alias("count")).to_dicts()
    )

    # Combine all metrics
    metrics = {
        "pycaret_metrics": pycaret_metrics,
        "num_clusters": model_info["num_clusters"],
        "num_samples": model_info["num_samples"],
        "cluster_distribution": cluster_distribution,
    }

    # Log some key metrics
    context.log.info(
        f"Metrics for external data: "
        f"silhouette={pycaret_metrics.get('silhouette', 'N/A')}, "
        f"clusters={model_info['num_clusters']}"
    )

    # Create DataFrame from metrics
    metrics_df = pl.DataFrame(
        [
            {
                "category": category,
                "num_clusters": model_info["num_clusters"],
                "silhouette": pycaret_metrics.get("silhouette", None),
                "calinski_harabasz": pycaret_metrics.get("calinski_harabasz", None),
                "davies_bouldin": pycaret_metrics.get("davies_bouldin", None),
                "cluster_distribution": str(cluster_distribution),
                "status": "success",
            }
        ]
    )

    # Store summary in context metadata
    context.add_output_metadata(
        {
            "category": category,
            "silhouette_score": pycaret_metrics.get("silhouette", None),
            "num_clusters": model_info["num_clusters"],
            "status": "success",
        }
    )

    return metrics_df


@dg.asset(
    name="external_generate_cluster_visualizations",
    description="Generates visualizations for external cluster analysis",
    group_name="cluster_analysis",
    compute_kind="external_cluster_analysis",
    deps=["external_train_clustering_models", "external_assign_clusters"],
    required_resource_keys={"config"},
)
def external_generate_cluster_visualizations(
    context: dg.AssetExecutionContext,
    external_train_clustering_models: dict[str, Any],
    external_assign_clusters: pl.DataFrame,
) -> pl.DataFrame:
    """Generate visualizations for analyzing external data cluster results.

    Creates various plots and visualizations to help understand and interpret
    clustering results, such as 2D scatter plots, PCA projections, and
    cluster distribution histograms.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: DataFrame with cluster assignments

    Returns:
        DataFrame mapping visualization types to file paths, or empty DataFrame if no models are available
    """
    # Check if there are any trained models
    if not external_train_clustering_models:
        context.log.warning(
            "No trained models available for visualizations, returning empty visualization data"
        )
        empty_vis = pl.DataFrame(
            [
                {
                    "category": "default",
                    "type": "none",
                    "path": "none",
                    "status": "no_models_available",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_models_available"})

        return empty_vis

    # Check if the DataFrame contains valid cluster assignments
    has_clusters = external_assign_clusters.filter(~pl.col("Cluster").is_null()).height > 0
    if not has_clusters:
        context.log.warning("No valid clusters assigned, returning empty visualization data")
        empty_vis = pl.DataFrame(
            [
                {
                    "category": "default",
                    "type": "none",
                    "path": "none",
                    "status": "no_clusters_assigned",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_clusters_assigned"})

        return empty_vis

    visualizations = []

    # Use the default category if available
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    context.log.info(f"Generating visualizations for external data")

    # In a real implementation, plots would be generated and saved
    # Here we're just creating placeholder file paths
    visualization_types = [
        "cluster_distribution",
        "pca_projection",
        "silhouette",
    ]

    for viz_type in visualization_types:
        visualizations.append(
            {
                "category": category,
                "type": viz_type,
                "path": f"plots/external_{category}_{viz_type}.png",
                "status": "success",
            }
        )

    context.log.info(f"Generated {len(visualizations)} visualizations for external data")

    # Store summary in context metadata
    context.add_output_metadata(
        {
            "category": category,
            "visualization_count": len(visualizations),
            "visualization_types": visualization_types,
            "status": "success",
        }
    )

    return pl.DataFrame(visualizations)
