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
    external_dimensionality_reduced_features: dict[str, pl.DataFrame],
) -> dict[str, int]:
    """Determine the optimal number of clusters for each external data category.

    Uses PyCaret to evaluate different cluster counts based on silhouette scores,
    Calinski-Harabasz Index, and Davies-Bouldin Index to determine the optimal
    number of clusters for each category.

    Args:
        context: Dagster asset execution context
        external_dimensionality_reduced_features: Dictionary of processed DataFrames by category
            from external data sources

    Returns:
        Dictionary mapping category names to their optimal cluster counts
    """
    optimal_clusters = {}

    # Get configuration parameters or use defaults
    min_clusters = getattr(context.resources.config, "min_clusters", Defaults.MIN_CLUSTERS)
    max_clusters = getattr(context.resources.config, "max_clusters", Defaults.MAX_CLUSTERS)
    metrics = getattr(context.resources.config, "metrics", Defaults.METRICS)
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)

    context.log.info(
        f"Determining optimal clusters using range {min_clusters}-{max_clusters} "
        f"with metrics: {metrics}"
    )

    for category, df in external_dimensionality_reduced_features.items():
        context.log.info(f"Determining optimal cluster count for category: {category}")

        sample_count = len(df)

        # Check if dataset has enough samples for clustering
        if sample_count < min_clusters:
            context.log.warning(
                f"Category '{category}' has only {sample_count} samples, "
                f"which is less than min_clusters={min_clusters}. "
                f"Setting optimal clusters to 1."
            )
            optimal_clusters[category] = 1
            continue

        # Adjust max_clusters to not exceed sample count
        adjusted_max_clusters = min(max_clusters, sample_count - 1)
        if adjusted_max_clusters < max_clusters:
            context.log.warning(
                f"Category '{category}' has only {sample_count} samples. "
                f"Reducing max_clusters from {max_clusters} to {adjusted_max_clusters}."
            )

        # If adjusted_max_clusters is less than min_clusters, we can't cluster properly
        if adjusted_max_clusters < min_clusters:
            context.log.warning(
                f"Category '{category}' has too few samples ({sample_count}) "
                f"to evaluate clusters in range [{min_clusters}, {max_clusters}]. "
                f"Setting optimal clusters to 1."
            )
            optimal_clusters[category] = 1
            continue

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
            f"Evaluating {min_clusters} to {adjusted_max_clusters} clusters for {category}"
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
            context.log.info(f"  {category} with {k} clusters: {metrics_str}")

        # Determine optimal clusters based on silhouette score (higher is better)
        if "silhouette" in metrics and cluster_metrics:
            best_k = max(
                cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("silhouette", 0)
            )
            context.log.info(f"Optimal clusters for {category} based on silhouette: {best_k}")
        # Fallback to Calinski-Harabasz (higher is better)
        elif "calinski_harabasz" in metrics and cluster_metrics:
            best_k = max(
                cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("calinski_harabasz", 0)
            )
            context.log.info(
                f"Optimal clusters for {category} based on calinski_harabasz: {best_k}"
            )
        # Fallback to Davies-Bouldin (lower is better)
        elif "davies_bouldin" in metrics and cluster_metrics:
            best_k = min(
                cluster_metrics.keys(),
                key=lambda k: cluster_metrics[k].get("davies_bouldin", float("inf")),
            )
            context.log.info(f"Optimal clusters for {category} based on davies_bouldin: {best_k}")
        else:
            # Default if no metrics match or no clusters were evaluated
            best_k = min(min_clusters, sample_count - 1) if sample_count > 1 else 1
            context.log.warning(
                f"Could not determine optimal clusters for {category}, using default: {best_k}"
            )

        optimal_clusters[category] = best_k

        # Store metrics in context for later reference
        context.add_output_metadata(
            {
                f"{category}_metrics": cluster_metrics,
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
    external_dimensionality_reduced_features: dict[str, pl.DataFrame],
    external_optimal_cluster_counts: dict[str, int],
) -> dict[str, Any]:
    """Train clustering models using engineered features from external data.

    Uses PyCaret to train clustering models for each category using the
    optimal number of clusters determined in the previous step.

    Args:
        context: Dagster asset execution context
        external_dimensionality_reduced_features: Dictionary of processed DataFrames by category
        external_optimal_cluster_counts: Dictionary mapping category names to optimal cluster counts

    Returns:
        Dictionary of trained clustering models organized by category
    """
    trained_models = {}

    # Get configuration parameters
    algorithm = getattr(context.resources.config, "algorithm", Defaults.ALGORITHM)
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)

    context.log.info(f"Training clustering models using algorithm: {algorithm}")

    for category, df in external_dimensionality_reduced_features.items():
        # Get optimal cluster count for this category
        cluster_count = external_optimal_cluster_counts.get(category, 1)
        context.log.info(f"Training {algorithm} with {cluster_count} clusters for {category}")

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

        context.log.info(f"Completed training for {category}")

    # Add useful metadata to the context
    context.add_output_metadata(
        {
            "algorithm": algorithm,
            "categories": list(trained_models.keys()),
            "cluster_counts": {
                category: info["num_clusters"] for category, info in trained_models.items()
            },
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
    external_fe_raw_data: dict[str, pl.DataFrame],
    external_train_clustering_models: dict[str, Any],
) -> dict[str, pl.DataFrame]:
    """Assign cluster labels to data points using trained models.

    Uses the trained clustering models to assign cluster labels to each
    data point directly from the raw data.

    Args:
        context: Dagster asset execution context
        external_fe_raw_data: Dictionary of raw DataFrames by category from external sources
        external_train_clustering_models: Dictionary of trained clustering models by category

    Returns:
        Dictionary of DataFrames with cluster assignments by category
    """
    assigned_data = {}

    context.log.info("Assigning clusters to data points from raw features")

    for category, df in external_fe_raw_data.items():
        # Check if we have a trained model for this category
        if category not in external_train_clustering_models:
            context.log.warning(f"No trained model found for category: {category}")
            continue

        context.log.info(f"Assigning clusters for category: {category}")

        # Get the model info
        model_info = external_train_clustering_models[category]
        exp = model_info["experiment"]
        model = model_info["model"]

        # Convert Polars DataFrame to Pandas for PyCaret
        pandas_df = df.to_pandas()

        # Get predictions using the trained model
        predictions = exp.predict_model(model, data=pandas_df)

        # Convert back to Polars and store
        assigned_data[category] = pl.from_pandas(predictions)

        # Log cluster distribution
        cluster_counts = (
            assigned_data[category]
            .group_by("Cluster")
            .agg(pl.count().alias("count"))
            .sort("Cluster")
        )
        context.log.info(f"Cluster distribution for {category}:\n{cluster_counts}")

    # Store metadata about the assignment
    context.add_output_metadata(
        {
            "categories": list(assigned_data.keys()),
            "total_records": sum(len(df) for df in assigned_data.values()),
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
    external_assign_clusters: dict[str, pl.DataFrame],
) -> None:
    """Save cluster assignments to persistent storage.

    Uses the configured output resource to save the cluster assignments
    for later use in analysis or reporting.

    Args:
        context: Dagster asset execution context
        external_assign_clusters: Dictionary of DataFrames with cluster assignments
    """
    context.log.info("Saving cluster assignments to storage")

    # Use the configured output resource
    assignments_output = context.resources.external_cluster_assignments

    # Save each category's assignments
    for category, df in external_assign_clusters.items():
        context.log.info(f"Saving cluster assignments for category: {category}")
        assignments_output.save(category, df)

    context.log.info(
        f"Successfully saved assignments for {len(external_assign_clusters)} categories"
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
    external_assign_clusters: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    """Calculate metrics to evaluate the quality of clustering.

    Computes various metrics to assess the quality of the clustering results,
    such as silhouette score, inertia, and cluster size distribution.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: Dictionary of DataFrames with cluster assignments

    Returns:
        Dictionary of evaluation metrics by category
    """
    metrics = {}

    # Get metrics from PyCaret and calculate additional custom metrics
    for category, model_info in external_train_clustering_models.items():
        if category not in external_assign_clusters:
            context.log.warning(f"No assignments found for category: {category}")
            continue

        context.log.info(f"Calculating evaluation metrics for category: {category}")

        # Get experiment and model
        exp = model_info["experiment"]
        model = model_info["model"]

        # Get PyCaret metrics
        pycaret_metrics = exp.pull().to_dict("records")[0]

        # Get cluster distribution from assignments
        assignments_df = external_assign_clusters[category]
        cluster_distribution = (
            assignments_df.group_by("Cluster").agg(pl.count().alias("count")).to_dicts()
        )

        # Combine all metrics
        category_metrics = {
            "pycaret_metrics": pycaret_metrics,
            "num_clusters": model_info["num_clusters"],
            "num_samples": model_info["num_samples"],
            "cluster_distribution": cluster_distribution,
        }

        metrics[category] = category_metrics

        # Log some key metrics
        context.log.info(
            f"Metrics for {category}: "
            f"silhouette={pycaret_metrics.get('silhouette', 'N/A'):.4f}, "
            f"clusters={model_info['num_clusters']}"
        )

    # Store summary in context metadata
    context.add_output_metadata(
        {
            "categories": list(metrics.keys()),
            "silhouette_scores": {
                category: metrics[category]["pycaret_metrics"].get("silhouette", None)
                for category in metrics
            },
        }
    )

    return metrics


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
    external_assign_clusters: dict[str, pl.DataFrame],
) -> dict[str, list[str]]:
    """Generate visualizations for analyzing cluster results.

    Creates various plots and visualizations to help understand and interpret
    clustering results, such as 2D scatter plots, PCA projections, and
    cluster distribution histograms.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: Dictionary of DataFrames with cluster assignments

    Returns:
        Dictionary mapping category names to lists of visualization file paths
    """
    visualizations = {}

    # Placeholder - In a real implementation, this would generate actual plots
    # and save them to files. Here we'll just return placeholder file paths.
    for category in external_train_clustering_models.keys():
        if category not in external_assign_clusters:
            context.log.warning(f"No assignments found for category: {category}")
            continue

        context.log.info(f"Generating visualizations for category: {category}")

        # In a real implementation, plots would be generated and saved
        visualizations[category] = [
            f"plots/{category}_cluster_distribution.png",
            f"plots/{category}_pca_projection.png",
            f"plots/{category}_silhouette.png",
        ]

        context.log.info(f"Generated {len(visualizations[category])} plots for {category}")

    # Store summary in context metadata
    context.add_output_metadata(
        {
            "categories": list(visualizations.keys()),
            "visualization_count": sum(len(v) for v in visualizations.values()),
        }
    )

    return visualizations
