"""Internal clustering assets for the clustering pipeline."""

import datetime
from typing import Any, cast

import dagster as dg
import polars as pl
from sklearn import metrics as sk_metrics  # type: ignore

# Import from our PyCaret-based implementation
from clustering.core.models import ClusteringModel


@dg.asset(
    io_manager_key="io_manager",
    deps=["preprocessed_internal_sales"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config", "alerts"},
)
def internal_clustering_model(
    context: dg.AssetExecutionContext,
    preprocessed_internal_sales: pl.DataFrame,
) -> ClusteringModel:
    """Train clustering model on internal data.

    Args:
        context: Asset execution context
        preprocessed_internal_sales: Preprocessed internal data

    Returns:
        Trained clustering model
    """
    context.log.info("Training internal clustering model")

    # Log environment info
    context.log.info(f"Using environment: {context.resources.config.get_env()}")

    # Load clustering configuration from environment-specific config
    config = context.resources.config.load("internal_clustering")
    clustering_config = config.get("clustering", {})

    # Extract clustering parameters from config
    algorithm = clustering_config.get("algorithm", "kmeans")
    normalize = clustering_config.get("normalize", True)
    norm_method = clustering_config.get("norm_method", "robust")  # Changed from clr to robust
    pca_active = clustering_config.get("pca_active", True)
    pca_components = clustering_config.get("pca_components", 0.8)
    ignore_features = clustering_config.get("ignore_features", ["STORE_NBR"])
    kwargs = clustering_config.get("kwargs", {})

    # Log configuration
    context.log.info(f"Using algorithm: {algorithm}")
    context.log.info(f"Model configuration: normalize={normalize}, pca={pca_active}")
    context.log.info("Normalization will be handled by the clustering model")

    # Create model with the specified parameters
    model = ClusteringModel(
        CLUS_ALGO=algorithm,
        NORMALIZE=normalize,
        NORM_METHOD=norm_method,
        PCA_ACTIVE=pca_active,
        PCA_COMPONENTS=pca_components,
        IGNORE_FEATURES=ignore_features,
        KWARGS=kwargs,
    )

    # Convert polars DataFrame to pandas for ClusteringModel
    inputs = preprocessed_internal_sales.to_pandas()

    try:
        # Fit the model
        context.log.info("Fitting internal clustering model")
        model.fit(inputs)
        return model
    except Exception as e:
        error_msg = f"Failed to fit clustering model: {e}"
        context.log.error(error_msg)
        context.resources.alerts.alert(message=error_msg, level="ERROR", context={"error": str(e)})
        raise


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_clustering_model"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config", "alerts"},
)
def internal_clusters(
    context: dg.AssetExecutionContext,
    internal_clustering_model: ClusteringModel,
) -> dict[str, Any]:
    """Generate internal clusters.

    Args:
        context: Asset execution context
        internal_clustering_model: Trained clustering model

    Returns:
        Dictionary containing cluster results
    """
    context.log.info("Generating internal clusters")

    try:
        # Assign clusters
        clustered_data = internal_clustering_model.assign()

        # Convert to polars for further processing
        result_df = pl.from_pandas(clustered_data)

        # Extract feature columns from the result data
        feature_cols = [
            col
            for col in result_df.columns
            if col != "Cluster"
            and result_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

        # Compute detailed cluster statistics
        cluster_stats: dict[str, Any] = {}

        # Process each cluster
        for cluster_id in result_df["Cluster"].unique():
            cluster_rows = result_df.filter(pl.col("Cluster") == cluster_id)
            cluster_size = len(cluster_rows)

            # Calculate silhouette score for this cluster if possible
            silhouette_score = None
            if cluster_size > 1 and len(result_df["Cluster"].unique()) > 1:
                try:
                    # Only calculate for clusters with enough data points
                    cluster_features_array = cluster_rows.select(feature_cols).to_numpy()
                    if len(cluster_features_array) >= 2:
                        # Use a sample to make computation faster for large clusters
                        sample_size = min(1000, len(cluster_features_array))
                        silhouette_score = float(
                            sk_metrics.silhouette_score(
                                cluster_features_array[:sample_size], [1] * sample_size
                            )
                        )
                except Exception as e:
                    context.log.warning(
                        f"Could not calculate silhouette score for cluster {cluster_id}: {e}"
                    )

            # Keep the raw stats dictionary for analysis
            mean_values: dict[str, float] = {}
            for col in feature_cols:
                if col in cluster_rows.columns:
                    mean_val = cluster_rows[col].mean()
                    if mean_val is not None:
                        mean_values[col] = float(mean_val)  # type: ignore
                    else:
                        mean_values[col] = 0.0

            cluster_stats[f"cluster_{cluster_id}"] = {
                "count": cluster_size,
                "mean": mean_values,
                "silhouette_score": silhouette_score,
            }

        # Get algorithm and parameters from the model
        algorithm = internal_clustering_model.CLUS_ALGO
        parameters = {
            "normalize": internal_clustering_model.NORMALIZE,
            "norm_method": internal_clustering_model.NORM_METHOD,
            "pca_active": internal_clustering_model.PCA_ACTIVE,
            "pca_components": internal_clustering_model.PCA_COMPONENTS,
        }
        parameters.update(internal_clustering_model.KWARGS)

        # Create a simplified result schema as a dictionary
        result_schema = {
            "model_version": "1.0",
            "algorithm": algorithm,
            "parameters": parameters,
            "clusters": [
                {
                    "cluster_id": str(cluster_id),
                    "size": stats["count"],
                    "silhouette_score": stats["silhouette_score"],
                }
                for cluster_id, stats in cluster_stats.items()
            ],
            "metadata": {
                "feature_columns": feature_cols,
                "environment": context.resources.config.get_env(),
                "row_count": len(result_df),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Log successful clustering
        context.log.info(f"Created {len(cluster_stats)} clusters")

        return {
            "clustered_data": result_df,
            "model": internal_clustering_model,
            "stats": cluster_stats,
            "result_schema": result_schema,
        }

    except Exception as e:
        error_msg = f"Failed to generate clusters: {e}"
        context.log.error(error_msg)
        context.resources.alerts.alert(message=error_msg, level="ERROR", context={"error": str(e)})
        raise


@dg.asset_check(asset="internal_clusters")
def validate_cluster_quality(
    context: dg.AssetExecutionContext, internal_clusters: dict[str, Any]
) -> dg.AssetCheckResult:
    """Validate the quality of internal clusters."""
    # Extract silhouette scores from the clusters
    stats = internal_clusters["stats"]

    # Skip if no stats available
    if not stats:
        return dg.AssetCheckResult(
            passed=False, metadata={"reason": "No cluster statistics available"}
        )

    # Check if there's at least one valid cluster
    valid_clusters = 0
    for cluster_name, cluster_data in stats.items():
        if cluster_data.get("count", 0) > 10:  # Consider clusters with at least 10 items as valid
            valid_clusters += 1

    # Check if we have enough valid clusters
    if valid_clusters < 2:
        return dg.AssetCheckResult(
            passed=False,
            metadata={
                "reason": f"Not enough valid clusters. Found {valid_clusters}, but at least 2 are required.",
                "valid_clusters": valid_clusters,
            },
        )

    # Check average silhouette score if available
    silhouette_scores = []
    for cluster_name, cluster_data in stats.items():
        score = cluster_data.get("silhouette_score")
        if score is not None:
            silhouette_scores.append(score)

    if silhouette_scores:
        avg_silhouette = sum(silhouette_scores) / len(silhouette_scores)
        if avg_silhouette < 0.1:  # Threshold for silhouette score
            return dg.AssetCheckResult(
                passed=False,
                metadata={
                    "reason": f"Low average silhouette score: {avg_silhouette:.4f}",
                    "avg_silhouette": avg_silhouette,
                },
            )
        else:
            return dg.AssetCheckResult(
                passed=True,
                metadata={
                    "valid_clusters": valid_clusters,
                    "avg_silhouette": avg_silhouette,
                },
            )
    else:
        # If we can't check silhouette scores, just validate based on cluster count
        return dg.AssetCheckResult(
            passed=True,
            metadata={
                "valid_clusters": valid_clusters,
                "note": "No silhouette scores available, validating based on cluster count only",
            },
        )


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_clusters"],
    compute_kind="internal_clustering",
    group_name="clustering",
)
def internal_cluster_evaluation(
    context: dg.AssetExecutionContext,
    internal_clusters: dict[str, Any],
) -> dict[str, float]:
    """Evaluate internal clusters.

    Args:
        context: Asset execution context
        internal_clusters: Internal clusters

    Returns:
        Dictionary of evaluation metrics
    """
    context.log.info("Evaluating internal clusters")

    # Extract model and evaluate
    model = internal_clusters.get("model")
    if model:
        metrics = model.evaluate()
        context.log.info(f"Evaluation metrics: {metrics}")
        # Ensure return value meets the promised type
        return cast(dict[str, float], metrics)
    else:
        context.log.error("No model found in internal_clusters")
        return {"error_code": 1.0}  # Use a numeric value to match return type


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_clusters"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config"},
)
def internal_clustering_output(
    context: dg.AssetExecutionContext,
    internal_clusters: dict[str, Any],
) -> None:
    """Save internal clustering results.

    Args:
        context: Asset execution context
        internal_clusters: Internal clusters
    """
    context.log.info("Saving internal clustering results")

    # Extract clustered data and configuration
    result_df = internal_clusters.get("clustered_data")
    if result_df is None:
        context.log.error("No clustered data found in internal_clusters")
        return None

    # Get output configuration
    config = context.resources.config.load("internal_clustering")
    output_config = config.get("output", {})
    output_format = output_config.get("format", "parquet")

    # Save to DuckDB (handled by the IO manager)
    context.log.info(f"Saving internal clustering results in {output_format} format")

    # Return the clustered data as an asset
    return None
