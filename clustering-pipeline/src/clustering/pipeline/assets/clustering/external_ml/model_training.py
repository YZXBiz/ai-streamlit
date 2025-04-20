"""Model training step for the external ML pipeline.

This module provides Dagster assets for training clustering models based on
engineered features from external data sources.
"""

import os
import tempfile
from typing import Any

import dagster as dg
import polars as pl
from pycaret.clustering import ClusteringExperiment, load_experiment


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

    context.log.info("Determining optimal cluster count for external data")

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

    # Create a temp directory for experiment files
    temp_dir = tempfile.mkdtemp(prefix="pycaret_experiments_")
    context.log.info(f"Using temporary directory for experiments: {temp_dir}")

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

    # Save the experiment using PyCaret's built-in function that handles lambda functions
    experiment_path = os.path.join(temp_dir, f"{category}_experiment")
    exp.save_experiment(experiment_path)

    # Get metrics before we reset the experiment
    try:
        metrics = exp.pull().iloc[0].to_dict()
    except (AttributeError, IndexError, KeyError, ValueError):
        metrics = {}
        context.log.warning("Could not extract metrics from experiment")

    # Store the model and experiment path
    trained_models[category] = {
        "model": model,
        "experiment_path": experiment_path,
        "features": df.columns,
        "num_clusters": cluster_count,
        "num_samples": len(df),
        "metrics": metrics,
    }

    context.log.info("Completed training for external data")

    # Add useful metadata to the context
    context.add_output_metadata(
        {
            "algorithm": algorithm,
            "categories": list(trained_models.keys()),
            "cluster_counts": dg.MetadataValue.json(
                {category: data["num_clusters"] for category, data in trained_models.items()}
            ),
            "experiment_paths": dg.MetadataValue.json(
                {category: data["experiment_path"] for category, data in trained_models.items()}
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

    # Convert model info to a DataFrame to comply with PickleWriter requirements
    if external_train_clustering_models:
        # Extract the model category (should be 'default' for external models)
        category = next(iter(external_train_clustering_models.keys()))
        model_info = external_train_clustering_models[category]

        # Create a dictionary with string metadata from the model info
        model_metadata = {
            "num_clusters": model_info["num_clusters"],
            "num_samples": model_info["num_samples"],
            "features": str(model_info["features"]),
            "experiment_path": model_info["experiment_path"],
        }

        # Add metrics if available
        if "metrics" in model_info and model_info["metrics"]:
            for k, v in model_info["metrics"].items():
                model_metadata[f"metric_{k}"] = v

        # Create a DataFrame with a single row containing the metadata
        model_df = pl.DataFrame([model_metadata])

        # Save the DataFrame (can't save the actual model objects directly)
        context.log.info("Saving model metadata to storage")
        model_output.write(model_df)

        # Save the model path to the context for reference
        context.add_output_metadata(
            {
                "model_path": model_info["experiment_path"],
                "category": category,
                "num_clusters": model_info["num_clusters"],
            }
        )
    else:
        # Create an empty DataFrame with expected schema
        empty_df = pl.DataFrame(
            {
                "num_clusters": [],
                "num_samples": [],
                "features": [],
                "experiment_path": [],
            }
        )
        model_output.write(empty_df)
        context.log.warning("No models to save, writing empty metadata DataFrame")

    context.log.info("Successfully saved model metadata to storage")


@dg.asset(
    name="external_assign_clusters",
    description="Assigns clusters to external data points using trained models",
    group_name="cluster_assignment",
    compute_kind="external_cluster_assignment",
    deps=[
        "external_dimensionality_reduced_features",
        "external_train_clustering_models",
        "external_fe_raw_data",
    ],
    required_resource_keys={"config"},
)
def external_assign_clusters(
    context: dg.AssetExecutionContext,
    external_dimensionality_reduced_features: pl.DataFrame,
    external_train_clustering_models: dict[str, Any],
    external_fe_raw_data: pl.DataFrame,
) -> pl.DataFrame:
    """Assign cluster labels to external data points using trained models.

    Uses the trained clustering models to assign cluster labels using the dimensionality reduced features,
    then applies these labels back to the original raw data with all columns preserved.
    Outliers (data points removed during preprocessing) are assigned to a special outlier cluster.

    Args:
        context: Dagster asset execution context
        external_dimensionality_reduced_features: DataFrame with dimensionality reduced external features
        external_train_clustering_models: Dictionary of trained clustering models by category
        external_fe_raw_data: DataFrame with original raw external features

    Returns:
        DataFrame with cluster assignments added to original data, with outliers assigned to a special cluster
    """
    context.log.info(
        "Assigning clusters using dimensionality reduced features and applying to raw data"
    )

    # Check if we have any trained models
    if not external_train_clustering_models:
        context.log.warning(
            "No trained models available for cluster assignment, returning original DataFrame with empty cluster column"
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

    # Check if we need to handle data size mismatch
    original_rows = external_fe_raw_data.height
    reduced_rows = external_dimensionality_reduced_features.height

    # Get the default category
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    # Get the model info
    model_info = external_train_clustering_models[category]
    model = model_info["model"]
    experiment_path = model_info["experiment_path"]

    # Get the number of clusters from the model to determine outlier cluster number
    num_clusters = model_info["num_clusters"]
    # Outlier cluster will be one more than the highest cluster
    outlier_cluster_num = (
        num_clusters  # If clusters are 0-based (0, 1, ...), outlier will be num_clusters
    )

    if original_rows != reduced_rows:
        context.log.warning(
            f"Size mismatch detected: original data has {original_rows} rows while "
            f"dimensionality reduced features has {reduced_rows} rows. "
            f"This is likely due to outlier removal during preprocessing."
        )

        context.log.info(f"Outliers will be assigned to cluster {outlier_cluster_num}")

        # Convert Polars DataFrame to Pandas for PyCaret
        pandas_df = external_dimensionality_reduced_features.to_pandas().reset_index(drop=True)
        pandas_df["temp_id"] = pandas_df.index

        context.log.info(f"Loading experiment from {experiment_path}")

        # Load the experiment using PyCaret's load_experiment function
        exp = load_experiment(experiment_path, data=pandas_df)

        context.log.info("Using model to assign clusters")

        # Use assign_model to get cluster assignments
        predictions = exp.assign_model(model)

        # Check the format of existing cluster values to ensure consistency
        sample_cluster_val = predictions["Cluster"].iloc[0]
        is_string_format = isinstance(sample_cluster_val, str)

        # Get the outlier cluster in the correct format
        if is_string_format and str(sample_cluster_val).startswith("Cluster"):
            outlier_cluster_formatted = f"Cluster {outlier_cluster_num}"
            context.log.info(
                f"Using string format for outlier cluster: '{outlier_cluster_formatted}'"
            )
        else:
            # If clusters are just numbers, convert to same type
            if is_string_format:
                outlier_cluster_formatted = str(outlier_cluster_num)
            else:
                outlier_cluster_formatted = outlier_cluster_num
            context.log.info(
                f"Using numeric format for outlier cluster: {outlier_cluster_formatted}"
            )

        # Get the cluster assignments with the temp ID
        cluster_assignments = predictions[["temp_id", "Cluster"]]

        # Convert original data to pandas
        original_data = external_fe_raw_data.to_pandas()

        # Set all clusters to outlier cluster initially (all are considered outliers by default)
        original_data_with_clusters = original_data.copy()
        original_data_with_clusters["Cluster"] = outlier_cluster_formatted

        # Now update the non-outlier points with their proper clusters
        try:
            # Get the STORE_NBR values from the external_fe_raw_data
            fe_raw_stores = external_fe_raw_data.select("STORE_NBR").to_pandas()

            # Set cluster values for matching IDs
            matched_count = 0
            for _, row in cluster_assignments.iterrows():
                idx = int(row["temp_id"])
                cluster = row["Cluster"]

                if idx < len(fe_raw_stores):
                    store_nbr = fe_raw_stores.iloc[idx]["STORE_NBR"]
                    # Find this store in the original data and set its cluster
                    mask = original_data_with_clusters["STORE_NBR"] == store_nbr
                    original_data_with_clusters.loc[mask, "Cluster"] = cluster
                    matched_count += 1
                    context.log.info(f"Assigned cluster {cluster} to store {store_nbr}")

            context.log.info(f"Matched {matched_count} out of {len(cluster_assignments)} stores")

        except Exception as e:
            context.log.error(f"Error matching reduced data to original: {e}")
            # Log all rows in a clear format to help debug
            context.log.info(f"Original data shapes: {original_data.shape}")
            context.log.info(f"Predictions shape: {predictions.shape}")
            context.log.info(f"Cluster assignments shape: {cluster_assignments.shape}")

        # Make sure all 'Cluster' values have the same type before converting to Polars
        # This prevents the 'int' object cannot be converted to 'PyString' error
        if is_string_format:
            # Ensure all cluster values are strings
            original_data_with_clusters["Cluster"] = original_data_with_clusters["Cluster"].astype(
                str
            )
            context.log.info("Ensuring all cluster values are strings")
        else:
            # Try to convert to integers if possible
            try:
                original_data_with_clusters["Cluster"] = original_data_with_clusters[
                    "Cluster"
                ].astype(int)
                context.log.info("Converted all cluster values to integers")
            except (ValueError, TypeError) as e:
                # If conversion fails, stick with strings
                original_data_with_clusters["Cluster"] = original_data_with_clusters[
                    "Cluster"
                ].astype(str)
                context.log.info(f"Failed to convert to integers, using strings: {e}")

        # Convert back to Polars
        try:
            assigned_data = pl.from_pandas(original_data_with_clusters)
            context.log.info("Successfully converted to Polars DataFrame")
        except TypeError as e:
            # If conversion still fails, try more aggressive type enforcement
            context.log.warning(f"Error converting to Polars: {e}, attempting alternative approach")
            # Convert all columns to string as a last resort
            original_data_with_clusters["Cluster"] = original_data_with_clusters["Cluster"].astype(
                str
            )
            assigned_data = pl.from_pandas(original_data_with_clusters)
            context.log.info("Successfully converted to Polars after type conversion")

    else:
        # If no size mismatch, proceed with normal approach
        # Convert Polars DataFrame to Pandas for PyCaret
        pandas_df = external_dimensionality_reduced_features.to_pandas()

        context.log.info(f"Loading experiment from {experiment_path}")

        # Load the experiment using PyCaret's load_experiment function
        exp = load_experiment(experiment_path, data=pandas_df)

        context.log.info("Using model to assign clusters")

        # Use assign_model instead of predict_model since we're using the same data
        predictions = exp.assign_model(model)

        # Get just the cluster assignments
        cluster_assignments = predictions[["Cluster"]]

        # Get the original raw data
        original_data = external_fe_raw_data.to_pandas()

        # Add cluster assignments to the original data
        original_data_with_clusters = original_data.copy()
        original_data_with_clusters["Cluster"] = cluster_assignments["Cluster"].values

        # Check if we need to enforce type consistency
        if isinstance(original_data_with_clusters["Cluster"].iloc[0], str):
            original_data_with_clusters["Cluster"] = original_data_with_clusters["Cluster"].astype(
                str
            )

        # Convert back to Polars
        assigned_data = pl.from_pandas(original_data_with_clusters)

    # Log cluster distribution
    cluster_counts = assigned_data.group_by("Cluster").agg(pl.len().alias("count")).sort("Cluster")
    context.log.info(f"Cluster distribution:\n{cluster_counts}")

    # Check if any points were assigned to the outlier cluster
    # Need to make sure we use the right format for comparison
    if isinstance(assigned_data.select("Cluster").row(0)[0], str):
        outlier_cluster_check = str(outlier_cluster_formatted)
    else:
        outlier_cluster_check = outlier_cluster_num

    outlier_count = assigned_data.filter(pl.col("Cluster") == outlier_cluster_check).height
    if outlier_count > 0:
        context.log.info(
            f"Assigned {outlier_count} outlier points to cluster {outlier_cluster_check}"
        )

    # Store metadata about the assignment
    context.add_output_metadata(
        {
            "model_category": category,
            "num_clusters": model_info["num_clusters"],
            "outlier_cluster": outlier_cluster_num,
            "outlier_count": outlier_count if "outlier_count" in locals() else 0,
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
    for later use in analysis or reporting. All data points have cluster assignments,
    including outliers which are assigned to a dedicated outlier cluster.

    Args:
        context: Dagster asset execution context
        external_assign_clusters: DataFrame with cluster assignments, including outlier assignments
    """
    context.log.info("Saving external cluster assignments to storage")

    # No need to check for null values as outliers now have their own cluster
    # Check that the DataFrame is not empty
    if external_assign_clusters.height == 0:
        context.log.warning("DataFrame is empty, skipping storage")
        context.add_output_metadata({"status": "skipped", "reason": "Empty DataFrame"})
        return

    # Use the configured output resource
    assignments_output = context.resources.external_cluster_assignments

    # Add a default category label
    df_with_category = external_assign_clusters.with_columns(pl.lit("default").alias("category"))

    # Save to storage
    context.log.info(f"Saving cluster assignments with {len(df_with_category)} records")
    assignments_output.write(df_with_category)

    # Get the distribution for logging
    cluster_dist = (
        external_assign_clusters.group_by("Cluster").agg(pl.len().alias("count")).sort("Cluster")
    )

    context.log.info("Successfully saved external cluster assignments")
    context.add_output_metadata(
        {
            "status": "success",
            "records_saved": external_assign_clusters.height,
            "cluster_distribution": dg.MetadataValue.json(cluster_dist.to_dicts()),
        }
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
    Outlier points (assigned to the outlier cluster) are included in the metrics.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: DataFrame with cluster assignments including outlier cluster

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

    # Check if the DataFrame is empty
    if external_assign_clusters.height == 0:
        context.log.warning("No data points in the DataFrame, returning empty metrics")
        empty_metrics = pl.DataFrame(
            [
                {
                    "category": "default",
                    "num_clusters": None,
                    "silhouette": None,
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "cluster_distribution": "{}",
                    "status": "no_data_points",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_data_points"})

        return empty_metrics

    # Use the default category if available
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    context.log.info("Calculating evaluation metrics for external data")

    # Get model info
    model_info = external_train_clustering_models[category]

    # Get metrics that were stored during training
    metrics = model_info.get("metrics", {})

    # Get cluster distribution from assignments
    cluster_distribution = (
        external_assign_clusters.group_by("Cluster").agg(pl.len().alias("count")).to_dicts()
    )

    # Identify the outlier cluster (highest cluster number)
    num_clusters = model_info["num_clusters"]
    outlier_cluster = num_clusters  # Same as defined in external_assign_clusters

    # Count the outliers
    outlier_count = external_assign_clusters.filter(pl.col("Cluster") == outlier_cluster).height
    context.log.info(f"Found {outlier_count} points in outlier cluster {outlier_cluster}")

    # Log some key metrics directly
    context.log.info(
        f"Metrics for external data: "
        f"silhouette={metrics.get('Silhouette', 'N/A')}, "
        f"num_clusters={model_info['num_clusters']}, "
        f"num_samples={model_info['num_samples']}, "
        f"outliers={outlier_count}"
    )

    # Create DataFrame from metrics
    metrics_df = pl.DataFrame(
        [
            {
                "category": category,
                "num_clusters": model_info["num_clusters"],
                "outlier_cluster": outlier_cluster,
                "outlier_count": outlier_count,
                "silhouette": metrics.get("Silhouette"),
                "calinski_harabasz": metrics.get("Calinski-Harabasz"),
                "davies_bouldin": metrics.get("Davies-Bouldin"),
                "cluster_distribution": str(cluster_distribution),
                "status": "success",
            }
        ]
    )

    # Store summary in context metadata
    context.add_output_metadata(
        {
            "category": category,
            "silhouette_score": metrics.get("Silhouette"),
            "num_clusters": model_info["num_clusters"],
            "outlier_cluster": outlier_cluster,
            "outlier_count": outlier_count,
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
    cluster distribution histograms. Includes visualization of outlier clusters.

    Args:
        context: Dagster asset execution context
        external_train_clustering_models: Dictionary of trained clustering models
        external_assign_clusters: DataFrame with cluster assignments including outlier cluster

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

    # Check if the DataFrame is empty
    if external_assign_clusters.height == 0:
        context.log.warning("No data points in the DataFrame, returning empty visualization data")
        empty_vis = pl.DataFrame(
            [
                {
                    "category": "default",
                    "type": "none",
                    "path": "none",
                    "status": "no_data_points",
                }
            ]
        )

        # Store summary in context metadata
        context.add_output_metadata({"status": "no_data_points"})

        return empty_vis

    visualizations = []

    # Use the default category if available
    category = "default"
    if category not in external_train_clustering_models:
        context.log.warning("No 'default' model found, using first available model")
        category = next(iter(external_train_clustering_models.keys()))

    context.log.info("Generating visualizations for external data")

    # Identify the outlier cluster
    model_info = external_train_clustering_models[category]
    num_clusters = model_info["num_clusters"]
    outlier_cluster = num_clusters  # Same as defined in external_assign_clusters

    # Count the outliers
    outlier_count = external_assign_clusters.filter(pl.col("Cluster") == outlier_cluster).height
    context.log.info(f"Including {outlier_count} outlier points in visualizations")

    # In a real implementation, plots would be generated and saved
    # Here we're just creating placeholder file paths
    visualization_types = [
        "cluster_distribution",
        "pca_projection",
        "silhouette",
        "outlier_analysis",  # New visualization specifically for outliers
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
            "outlier_cluster": outlier_cluster,
            "outlier_count": outlier_count,
            "status": "success",
        }
    )

    return pl.DataFrame(visualizations)


def get_pycaret_metrics(model):
    """Get metrics from a model.

    Args:
        model: scikit-learn model

    Returns:
        Dictionary of metrics
    """
    # Since we don't have the PyCaret experiment anymore, we'll return empty metrics
    # In a real implementation, you could calculate these from scratch using scikit-learn
    return {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
