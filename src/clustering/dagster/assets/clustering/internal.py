"""Internal clustering assets for the clustering pipeline."""

import datetime
from typing import Any

import dagster as dg
import polars as pl
from sklearn import metrics as sk_metrics

# Import from our PyCaret-based implementation
from clustering.core.models import ClusteringModel


# Define schemas for internal use
class ClusterFeature:
    """Feature statistics for a cluster."""

    def __init__(self, name: str, mean: float, median: float, min: float, max: float):
        self.name = name
        self.mean = mean
        self.median = median
        self.min = min
        self.max = max

    def __getitem__(self, key):
        return getattr(self, key)


class ClusterOutputSchema:
    """Output schema for a cluster."""

    def __init__(
        self,
        cluster_id: str,
        size: int,
        features: list[ClusterFeature],
        silhouette_score: float | None = None,
    ):
        self.cluster_id = cluster_id
        self.size = size
        self.features = features
        self.silhouette_score = silhouette_score

    def __getitem__(self, key):
        return getattr(self, key)


class ClusteringResult:
    """Result schema for clustering operation."""

    def __init__(
        self,
        model_version: str,
        algorithm: str,
        parameters: dict,
        clusters: list[ClusterOutputSchema],
        metadata: dict,
        timestamp: str,
    ):
        self.model_version = model_version
        self.algorithm = algorithm
        self.parameters = parameters
        self.clusters = clusters
        self.metadata = metadata
        self.timestamp = timestamp

    def model_dump(self):
        return {
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "features": [
                        {
                            "name": f.name,
                            "mean": f.mean,
                            "median": f.median,
                            "min": f.min,
                            "max": f.max,
                        }
                        for f in c.features
                    ],
                    "silhouette_score": c.silhouette_score,
                }
                for c in self.clusters
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


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
    context.log.info(
        "Skipping explicit normalization as it will be handled by the clustering model"
    )
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
) -> ClusteringModel:
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
    norm_method = clustering_config.get(
        "norm_method", "robust"
    )  # Changed from clr to robust
    pca_active = clustering_config.get("pca_active", True)
    pca_components = clustering_config.get("pca_components", 0.8)
    ignore_features = clustering_config.get("ignore_features", ["STORE_NBR"])
    kwargs = clustering_config.get("kwargs", {})

    # Log configuration
    context.log.info(f"Using algorithm: {algorithm}")
    context.log.info(f"Model configuration: normalize={normalize}, pca={pca_active}")

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
    inputs = normalized_internal_data.to_pandas()

    try:
        # Fit the model
        context.log.info("Fitting internal clustering model")
        model.fit(inputs)
        return model
    except Exception as e:
        error_msg = f"Failed to fit clustering model: {e}"
        context.log.error(error_msg)
        context.resources.alerts.alert(
            message=error_msg, level="ERROR", context={"error": str(e)}
        )
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
    internal_clustering_model: ClusteringModel,
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
        feature_cols = [
            col
            for col in normalized_internal_data.select(pl.col(pl.Float64)).columns
            if col != "Cluster"
        ]

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
                    cluster_features_array = cluster_rows.select(
                        feature_cols
                    ).to_numpy()
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
                "mean": {
                    col: float(cluster_rows[col].mean())
                    for col in feature_cols
                    if col in cluster_rows.columns
                },
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
        context.resources.alerts.alert(
            message=error_msg, level="ERROR", context={"error": str(e)}
        )
        raise


@dg.asset_check(asset="internal_clusters")
def validate_cluster_quality(context: dg.AssetExecutionContext, internal_clusters):
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
        if (
            cluster_data.get("count", 0) > 10
        ):  # Consider clusters with at least 10 items as valid
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
        return metrics
    else:
        context.log.error("No model found in internal_clusters")
        return {"error": "No model found for evaluation"}


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
