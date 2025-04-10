"""Model training step for the internal ML pipeline.

This module provides Dagster assets for training clustering models based on
engineered features.
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
    name="optimal_cluster_counts",
    description="Determines optimal number of clusters for each category",
    group_name="model_training",
    compute_kind="training",
    deps=["dimensionality_reduced_features"],
    required_resource_keys={"config"},
)
def optimal_cluster_counts(
    context: dg.AssetExecutionContext,
    dimensionality_reduced_features: dict[str, pl.DataFrame],
) -> dict[str, int]:
    """Determine the optimal number of clusters for each category.

    Uses PyCaret to evaluate different cluster counts based on silhouette scores,
    Calinski-Harabasz Index, and Davies-Bouldin Index to determine the optimal
    number of clusters for each category.

    Args:
        context: Dagster asset execution context
        dimensionality_reduced_features: Dictionary of processed DataFrames by category

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

    for category, df in dimensionality_reduced_features.items():
        context.log.info(f"Determining optimal cluster count for category: {category}")

        # Check if dataset has enough samples for clustering
        if len(df) < min_clusters:
            context.log.warning(
                f"Category '{category}' has only {len(df)} samples, "
                f"which is less than min_clusters={min_clusters}. "
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
        context.log.info(f"Evaluating {min_clusters} to {max_clusters} clusters for {category}")
        cluster_metrics = {}

        # Track metric values for each cluster count
        for k in range(min_clusters, max_clusters + 1):
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
        if "silhouette" in metrics:
            best_k = max(
                cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("silhouette", 0)
            )
            context.log.info(f"Optimal clusters for {category} based on silhouette: {best_k}")
        # Fallback to Calinski-Harabasz (higher is better)
        elif "calinski_harabasz" in metrics:
            best_k = max(
                cluster_metrics.keys(), key=lambda k: cluster_metrics[k].get("calinski_harabasz", 0)
            )
            context.log.info(
                f"Optimal clusters for {category} based on calinski_harabasz: {best_k}"
            )
        # Fallback to Davies-Bouldin (lower is better)
        elif "davies_bouldin" in metrics:
            best_k = min(
                cluster_metrics.keys(),
                key=lambda k: cluster_metrics[k].get("davies_bouldin", float("inf")),
            )
            context.log.info(f"Optimal clusters for {category} based on davies_bouldin: {best_k}")
        else:
            # Default if no metrics match
            best_k = (min_clusters + max_clusters) // 2
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
    name="trained_clustering_models",
    description="Trains clustering models using optimal number of clusters",
    group_name="model_training",
    compute_kind="training",
    deps=["dimensionality_reduced_features", "optimal_cluster_counts"],
    required_resource_keys={"config"},
)
def train_clustering_models(
    context: dg.AssetExecutionContext,
    dimensionality_reduced_features: dict[str, pl.DataFrame],
    optimal_cluster_counts: dict[str, int],
) -> dict[str, Any]:
    """Train clustering models using engineered features.

    Uses PyCaret to train clustering models for each category using the
    optimal number of clusters determined in the previous step.

    Args:
        context: Dagster asset execution context
        dimensionality_reduced_features: Dictionary of processed DataFrames by category
        optimal_cluster_counts: Dictionary mapping category names to optimal cluster counts

    Returns:
        Dictionary containing trained models and metadata
    """
    models = {}
    models_info = {}

    # Get configuration parameters or use defaults
    algorithm = getattr(context.resources.config, "algorithm", Defaults.ALGORITHM)
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)

    context.log.info(f"Using clustering algorithm: {algorithm}")

    for category, df in dimensionality_reduced_features.items():
        # Get optimal cluster count for this category
        num_clusters = optimal_cluster_counts.get(category, Defaults.MIN_CLUSTERS)

        # Skip categories with too few samples for meaningful clustering
        if num_clusters < 2:
            context.log.warning(
                f"Category '{category}' has an optimal cluster count of {num_clusters}, "
                f"which is less than 2. Skipping model training for this category."
            )
            models_info[category] = {
                "algorithm": algorithm,
                "num_clusters": num_clusters,
                "metrics": {},
                "status": "skipped",
                "reason": "insufficient_data_for_clustering",
            }
            continue

        context.log.info(f"Training {algorithm} model for {category} with {num_clusters} clusters")

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment
        exp = ClusteringExperiment()
        exp.setup(
            data=pandas_df,
            session_id=session_id,
            verbose=False,
        )

        # Train the model with optimal clusters
        model = exp.create_model(algorithm, num_clusters=num_clusters, verbose=False)

        # Extract metrics
        metrics = exp.pull()

        # Store model and metadata
        models[category] = model

        # Store information about the training
        models_info[category] = {
            "algorithm": algorithm,
            "num_clusters": num_clusters,
            "metrics": metrics.to_dict(orient="records")[0],
            "status": "success",
        }

        # Log results
        context.log.info(f"Model training completed for {category}")

    # Add metadata for tracking
    context.add_output_metadata(
        {
            "models_trained": list(models.keys()),
            "algorithm_used": algorithm,
        }
    )

    # Return both models and their metadata
    return {
        "models": models,
        "info": models_info,
    }


@dg.asset(
    name="saved_clustering_models",
    description="Persists trained clustering models to storage",
    group_name="model_training",
    compute_kind="io",
    deps=["trained_clustering_models"],
    required_resource_keys={"config", "model_output"},
)
def save_clustering_models(
    context: dg.AssetExecutionContext,
    trained_clustering_models: dict[str, Any],
) -> None:
    """Save trained clustering models to the configured output location.

    Args:
        context: Dagster asset execution context
        trained_clustering_models: Dictionary containing trained models and metadata

    Returns:
        None
    """
    models = trained_clustering_models.get("models", {})

    # Save models using the configured writer - only if we have models
    if models:
        context.log.info("Saving trained models to output location")
        context.resources.model_output.write(models)
        return True
    else:
        context.log.warning(
            "No models were trained (all categories were skipped or had insufficient data). "
            "Skipping model output write."
        )
        return False


@dg.asset(
    name="cluster_assignments",
    description="Assigns clusters to data points using trained models",
    group_name="cluster_prediction",
    compute_kind="prediction",
    deps=["dimensionality_reduced_features", "trained_clustering_models"],
    required_resource_keys={"config"},
)
def assign_clusters(
    context: dg.AssetExecutionContext,
    dimensionality_reduced_features: dict[str, pl.DataFrame],
    trained_clustering_models: dict[str, Any],
) -> dict[str, pl.DataFrame]:
    """Assign clusters to data points using trained models.

    Uses PyCaret's assign_model function to add cluster labels to the original data,
    preserving all features including those that might have been ignored during training.

    Args:
        context: Dagster asset execution context
        dimensionality_reduced_features: Dictionary of processed DataFrames by category
        trained_clustering_models: Dictionary containing trained models and metadata

    Returns:
        Dictionary of DataFrames with cluster assignments added
    """
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)
    ignored_features = getattr(context.resources.config, "ignore_features", [])

    if ignored_features:
        context.log.info(
            f"Note: Previously ignored features {ignored_features} will be included in output"
        )

    models = trained_clustering_models.get("models", {})
    model_info = trained_clustering_models.get("info", {})
    all_clustered_data = {}

    for category, df in dimensionality_reduced_features.items():
        # Check if we have a model for this category
        if category not in models:
            # If category was skipped during model training
            if category in model_info and model_info[category].get("status") == "skipped":
                context.log.info(f"No model available for {category}, using default cluster 0")
                # Create a dummy clustered dataset with all points in cluster 0
                all_clustered_data[category] = df.with_columns(pl.lit(0).alias("Cluster"))
                continue
            else:
                context.log.warning(f"No model found for category {category}, skipping")
                continue

        # Get the model for this category
        model = models[category]

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment and assign clusters
        exp = ClusteringExperiment()
        exp.setup(
            data=pandas_df,
            session_id=session_id,
            verbose=False,
        )

        # Use assign_model to get cluster assignments - this preserves all original features
        clustered_data = exp.assign_model(model)

        # Convert back to Polars and store
        all_clustered_data[category] = pl.from_pandas(clustered_data)

        # Log cluster distribution
        cluster_counts = clustered_data["Cluster"].value_counts().to_dict()
        context.log.info(f"Cluster distribution for {category}: {cluster_counts}")

    return all_clustered_data


@dg.asset(
    name="persisted_cluster_assignments",
    description="Saves cluster assignments to storage",
    group_name="cluster_prediction",
    compute_kind="io",
    deps=["cluster_assignments"],
    required_resource_keys={"cluster_assignments"},
)
def save_cluster_assignments(
    context: dg.AssetExecutionContext,
    cluster_assignments: dict[str, pl.DataFrame],
) -> None:
    """Save cluster assignments to the configured output location.

    Args:
        context: Dagster asset execution context
        cluster_assignments: Dictionary of DataFrames with cluster assignments

    Returns:
        None
    """
    # Save cluster assignments
    if cluster_assignments:
        context.log.info("Saving cluster assignments to output location")
        context.resources.cluster_assignments.write(cluster_assignments)
        return True
    else:
        context.log.warning("No cluster assignments to save")
        return False


@dg.asset(
    name="cluster_metrics",
    description="Calculates metrics for cluster quality evaluation",
    group_name="model_analysis",
    compute_kind="evaluation",
    deps=["trained_clustering_models", "cluster_assignments"],
    required_resource_keys={"config"},
)
def calculate_cluster_metrics(
    context: dg.AssetExecutionContext,
    trained_clustering_models: dict[str, Any],
    cluster_assignments: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    """Calculate metrics to evaluate clustering quality.

    Extracts and organizes clustering quality metrics for each category.

    Args:
        context: Dagster asset execution context
        trained_clustering_models: Dictionary containing trained models and metadata
        cluster_assignments: Dictionary of DataFrames with cluster assignments

    Returns:
        Dictionary of evaluation metrics by category
    """
    model_info = trained_clustering_models.get("info", {})
    evaluation_metrics = {}

    for category in model_info.keys():
        # Skip categories that don't have data
        if category not in cluster_assignments:
            context.log.warning(f"No cluster assignments found for {category}, skipping evaluation")
            continue

        context.log.info(f"Extracting clustering metrics for {category}")

        # Get model metrics from model_info
        if category in model_info:
            metrics = model_info[category].get("metrics", {})
        else:
            metrics = {}

        # Get the cluster assignments for this category
        df = cluster_assignments[category]

        # Get cluster distribution
        if "Cluster" in df.columns:
            cluster_counts = df.group_by("Cluster").agg(pl.count()).to_dicts()
        else:
            cluster_counts = []

        # Store metrics for this category
        evaluation_metrics[category] = {
            "quality_metrics": metrics,
            "cluster_distribution": cluster_counts,
        }

        # Add evaluation metrics to metadata
        context.add_output_metadata(
            {
                f"{category}_metrics": {
                    "quality_metrics": metrics,
                    "cluster_distribution": cluster_counts,
                }
            }
        )

    return evaluation_metrics


@dg.asset(
    name="cluster_visualizations",
    description="Generates visualizations for cluster analysis",
    group_name="model_analysis",
    compute_kind="visualization",
    deps=["trained_clustering_models", "cluster_assignments"],
    required_resource_keys={"config"},
)
def generate_cluster_visualizations(
    context: dg.AssetExecutionContext,
    trained_clustering_models: dict[str, Any],
    cluster_assignments: dict[str, pl.DataFrame],
) -> dict[str, list[str]]:
    """Generate visualizations for cluster analysis.

    Creates various plots to help analyze and understand the clustering results.

    Args:
        context: Dagster asset execution context
        trained_clustering_models: Dictionary containing trained models and metadata
        cluster_assignments: Dictionary of DataFrames with cluster assignments

    Returns:
        Dictionary mapping categories to lists of generated visualization types
    """
    session_id = getattr(context.resources.config, "session_id", Defaults.SESSION_ID)
    plot_types = ["elbow", "silhouette", "distance", "distribution"]

    models = trained_clustering_models.get("models", {})
    model_info = trained_clustering_models.get("info", {})

    visualization_results = {}

    for category in models.keys():
        model = models[category]

        # Skip categories that don't have data
        if category not in cluster_assignments:
            context.log.warning(
                f"No cluster assignments found for {category}, skipping visualization"
            )
            continue

        context.log.info(f"Generating visualizations for {category}")

        # Get the cluster assignments for this category
        df = cluster_assignments[category]

        # Initialize results list for this category
        category_visualizations = []

        # Generate plots if we have a valid model
        if model is not None:
            # Convert to pandas for PyCaret
            pandas_df = df.to_pandas()

            # Initialize PyCaret experiment
            exp = ClusteringExperiment()
            exp.setup(
                data=pandas_df,
                session_id=session_id,
                verbose=False,
            )

            # Load the model into the experiment
            loaded_model = exp.create_model(
                estimator=type(model).__name__,
                num_clusters=model_info[category].get("num_clusters", 2),
                verbose=False,
            )

            # Generate plots
            for plot_type in plot_types:
                try:
                    context.log.info(f"Generating {plot_type} plot for {category}")

                    # Generate the plot
                    _ = exp.plot_model(loaded_model, plot=plot_type, verbose=False)

                    # Record that we generated this plot
                    category_visualizations.append(plot_type)

                    # In a real implementation, we would save the plot to a file or database
                    # We could also use Dagster's built-in plotting capabilities

                except Exception as e:
                    context.log.error(f"Error generating {plot_type} plot for {category}: {str(e)}")

        # Store the results for this category
        visualization_results[category] = category_visualizations

        # Add visualization results to metadata
        context.add_output_metadata(
            {
                f"{category}_visualizations": category_visualizations,
            }
        )

    return visualization_results
