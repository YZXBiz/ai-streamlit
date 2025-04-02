"""Internal clustering assets for the clustering pipeline."""

import datetime
from typing import Any

import dagster as dg
import polars as pl
from sklearn import metrics as sk_metrics

from clustering.core import models
from clustering.core.schemas import ClusterFeature, ClusteringResult, ClusterOutputSchema


@dg.asset(
    io_manager_key="io_manager",
    deps=["preprocessed_internal_sales"],
    compute_kind="internal_clustering",
    group_name="clustering",
)
def normalized_internal_data(
    context: dg.AssetExecutionContext,
    preprocessed_internal_sales: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize internal data for clustering.

    Args:
        context: Asset execution context
        preprocessed_internal_sales: Preprocessed sales data

    Returns:
        Normalized data for clustering
    """
    context.log.info("Normalizing internal data")

    # Log environment info
    context.log.info(f"Using environment: {context.resources.config.get_env()}")

    # We don't need to normalize here since the ClusteringModel handles normalization
    # Just return the preprocessed data
    context.log.info("Skipping explicit normalization as it will be handled by the clustering model")
    return preprocessed_internal_sales


@dg.asset(
    io_manager_key="io_manager",
    deps=["normalized_internal_data"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config", "alerts"},
)
def internal_clustering_model(
    context: dg.AssetExecutionContext,
    normalized_internal_data: pl.DataFrame,
) -> models.ClusteringModel:
    """Train clustering model on internal data.

    Args:
        context: Asset execution context
        normalized_internal_data: Normalized internal data

    Returns:
        Trained clustering model
    """
    context.log.info("Training internal clustering model")

    # Load clustering configuration from environment-specific config
    config = context.resources.config.load("internal_clustering")
    clustering_config = config.get("clustering", {})

    # Extract clustering parameters from config
    algorithm = clustering_config.get("algorithm", "kmeans")
    normalize = clustering_config.get("normalize", True)
    norm_method = clustering_config.get("norm_method", "clr")
    pca_active = clustering_config.get("pca_active", True)
    pca_components = clustering_config.get("pca_components", 0.8)
    ignore_features = clustering_config.get("ignore_features", ["STORE_NBR"])
    kwargs = clustering_config.get("kwargs", {})

    # Log configuration
    context.log.info(f"Using algorithm: {algorithm}")
    context.log.info(f"Model configuration: normalize={normalize}, pca={pca_active}")

    # Create model with the specified parameters
    model = models.ClusteringModel(
        CLUS_ALGO=algorithm,
        NORMALIZE=normalize,
        NORM_METHOD=norm_method,
        PCA_ACTIVE=pca_active,
        PCA_COMPONENTS=pca_components,
        IGNORE_FEATURES=ignore_features,
        KWARGS=kwargs,
    )

    # Convert polars DataFrame to pandas for ClusteringModel
    inputs = normalized_internal_data.to_pandas()

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
    deps=["internal_clustering_model", "normalized_internal_data"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config", "alerts"},
)
def internal_clusters(
    context: dg.AssetExecutionContext,
    internal_clustering_model: models.ClusteringModel,
    normalized_internal_data: pl.DataFrame,
) -> dict[str, Any]:
    """Generate internal clusters.

    Args:
        context: Asset execution context
        internal_clustering_model: Trained clustering model
        normalized_internal_data: Normalized data for clustering

    Returns:
        Dictionary containing cluster results
    """
    context.log.info("Generating internal clusters")

    try:
        # Assign clusters
        clustered_data = internal_clustering_model.assign()

        # Convert to polars for further processing
        result_df = pl.from_pandas(clustered_data)

        # Extract feature columns (numeric columns excluding the cluster column)
        feature_cols = [col for col in normalized_internal_data.select(pl.col(pl.Float64)).columns if col != "Cluster"]

        # Compute detailed cluster statistics
        cluster_stats = {}
        cluster_schemas = []

        # Process each cluster
        for cluster_id in result_df["Cluster"].unique():
            cluster_rows = result_df.filter(pl.col("Cluster") == cluster_id)
            cluster_size = len(cluster_rows)

            # Compute feature statistics for the cluster
            cluster_features = []
            for col in feature_cols:
                if col in cluster_rows.columns:
                    feature_stats = {
                        "name": col,
                        "mean": float(cluster_rows[col].mean()),
                        "median": float(cluster_rows[col].median()),
                        "min": float(cluster_rows[col].min()),
                        "max": float(cluster_rows[col].max()),
                    }
                    cluster_features.append(ClusterFeature(**feature_stats))

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
                            sk_metrics.silhouette_score(cluster_features_array[:sample_size], [1] * sample_size)
                        )
                except Exception as e:
                    context.log.warning(f"Could not calculate silhouette score for cluster {cluster_id}: {e}")

            # Create a validated cluster schema
            cluster_schema = ClusterOutputSchema(
                cluster_id=str(cluster_id),
                size=cluster_size,
                features=cluster_features,
                silhouette_score=silhouette_score,
            )

            # Store the validated cluster schema
            cluster_schemas.append(cluster_schema)

            # Also keep the raw stats dictionary for backward compatibility
            cluster_stats[f"cluster_{cluster_id}"] = {
                "count": cluster_size,
                "mean": {col: float(cluster_rows[col].mean()) for col in feature_cols if col in cluster_rows.columns},
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

        # Validate and create the full clustering result
        clustering_result = ClusteringResult(
            model_version="1.0",
            algorithm=algorithm,
            parameters=parameters,
            clusters=cluster_schemas,
            metadata={
                "feature_columns": feature_cols,
                "environment": context.resources.config.get_env(),
                "row_count": len(normalized_internal_data),
            },
            timestamp=datetime.datetime.now().isoformat(),
        )

        # Log successful clustering
        context.log.info(f"Created {len(cluster_schemas)} clusters")

        return {
            "clustered_data": result_df,
            "model": internal_clustering_model,
            "stats": cluster_stats,
            "result_schema": clustering_result.model_dump(),
        }

    except Exception as e:
        error_msg = f"Failed to generate clusters: {e}"
        context.log.error(error_msg)
        context.resources.alerts.alert(message=error_msg, level="ERROR", context={"error": str(e)})
        raise


@dg.asset_check(asset="internal_clusters", severity=dg.AssetCheckSeverity.ERROR)
def validate_cluster_quality(context: dg.AssetExecutionContext, internal_clusters):
    """Validate the quality of internal clusters."""
    # Extract silhouette scores from the clusters
    stats = internal_clusters["stats"]

    # Skip if no stats available
    if not stats:
        return dg.AssetCheckResult(passed=False, metadata={"reason": "No cluster statistics available"})

    # Check if there's at least one valid cluster
    valid_clusters = 0
    for cluster_name, cluster_info in stats.items():
        if "count" in cluster_info and cluster_info["count"] > 1:
            valid_clusters += 1

    if valid_clusters == 0:
        return dg.AssetCheckResult(passed=False, metadata={"reason": "No valid clusters found"})

    # Check for clusters that are too small
    clusters_with_issues = []
    for cluster_name, cluster_info in stats.items():
        if cluster_info.get("count", 0) < 3:  # Minimum meaningful cluster size
            clusters_with_issues.append(f"{cluster_name}: size={cluster_info.get('count', 0)}")

    # Get overall silhouette score if available
    overall_score = None
    try:
        if "clustered_data" in internal_clusters and "model" in internal_clusters:
            clustered_data = internal_clusters["clustered_data"]
            feature_cols = [
                col
                for col in clustered_data.columns
                if col not in ["Cluster"] and clustered_data[col].dtype in [pl.Float32, pl.Float64]
            ]

            if feature_cols and "Cluster" in clustered_data.columns:
                features = clustered_data.select(feature_cols).to_numpy()
                labels = clustered_data["Cluster"].to_numpy()

                # Skip if we have only one cluster
                if len(set(labels)) > 1:
                    # Sample if dataset is large
                    max_samples = 10000
                    if len(features) > max_samples:
                        import numpy as np

                        indices = np.random.choice(len(features), max_samples, replace=False)
                        features = features[indices]
                        labels = labels[indices]

                    overall_score = float(sk_metrics.silhouette_score(features, labels))

                    # Check if score is too low
                    if overall_score < 0.2:  # This threshold can be adjusted
                        return dg.AssetCheckResult(
                            passed=False,
                            metadata={
                                "reason": "Low silhouette score indicating poor cluster separation",
                                "silhouette_score": overall_score,
                                "clusters_with_issues": clusters_with_issues,
                            },
                        )
    except Exception as e:
        context.log.warning(f"Could not calculate overall silhouette score: {e}")

    # If we have clusters with issues, report them but don't fail
    if clusters_with_issues:
        return dg.AssetCheckResult(
            passed=True,
            metadata={
                "warning": "Some clusters are very small",
                "clusters_with_issues": clusters_with_issues,
                "overall_silhouette_score": overall_score,
            },
        )

    return dg.AssetCheckResult(passed=True, metadata={"overall_silhouette_score": overall_score})


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
    """Evaluate internal clustering quality.

    Args:
        context: Asset execution context
        internal_clusters: Internal clusters data

    Returns:
        Dictionary of evaluation metrics
    """
    context.log.info("Evaluating internal clusters")

    # Extract the model and get evaluation results
    model = internal_clusters["model"]

    # Use the model's evaluate method
    scores = model.evaluate()

    # Convert to dictionary if it's a DataFrame
    if hasattr(scores, "to_dict"):
        scores_dict = scores.to_dict()
    else:
        scores_dict = scores

    return scores_dict


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
        internal_clusters: Internal clusters data
    """
    context.log.info("Saving internal clustering results")

    # Get clustered data
    clustered_data = internal_clusters["clustered_data"]
    result_schema = internal_clusters["result_schema"]

    # Save the model
    model = internal_clusters["model"]
    model.save()

    # Example: Save results to configured location (assuming config has output_path)
    config = context.resources.config.load("internal_clustering")
    output_path = config.get("output", {}).get("path", "output/internal_clustering_results.csv")

    # Save the clustered data
    if isinstance(clustered_data, pl.DataFrame):
        # Use appropriate method based on file extension
        if output_path.endswith(".csv"):
            clustered_data.write_csv(output_path)
        elif output_path.endswith(".parquet"):
            clustered_data.write_parquet(output_path)
        else:
            # Default to CSV
            clustered_data.write_csv(output_path)

    context.log.info(f"Saved clustering results to {output_path}")

    # Example: Also save the schema result as JSON
    import json

    schema_path = output_path.rsplit(".", 1)[0] + "_schema.json"
    with open(schema_path, "w") as f:
        json.dump(result_schema, f, indent=2)

    context.log.info(f"Saved cluster schema to {schema_path}")

    return None
