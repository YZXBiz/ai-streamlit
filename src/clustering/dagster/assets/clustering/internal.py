"""Internal clustering assets for the clustering pipeline."""

import datetime
from typing import Any, Dict

import dagster as dg
import polars as pl
from clustpy.algorithms import hdbscan, kmeans
from sklearn import metrics as sk_metrics

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

    # Load configuration
    config = context.resources.config.load("internal_clustering")
    context.log.info(f"Using environment: {context.resources.config.get_env()}")

    # Perform normalization (this would use your existing normalization logic)
    # For example:
    numeric_cols = preprocessed_internal_sales.select(pl.col(pl.Float64)).columns

    # Basic standardization - Replace with your actual normalization logic
    normalized_data = preprocessed_internal_sales.clone()
    for col in numeric_cols:
        mean = preprocessed_internal_sales[col].mean()
        std = preprocessed_internal_sales[col].std()
        if std > 0:
            normalized_data = normalized_data.with_columns(
                pl.col(col).map_elements(lambda x: (x - mean) / std).alias(col)
            )

    return normalized_data


@dg.asset(
    io_manager_key="io_manager",
    deps=["normalized_internal_data"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config", "alerts"},
)
def internal_clusters(
    context: dg.AssetExecutionContext,
    normalized_internal_data: pl.DataFrame,
) -> Dict[str, Any]:
    """Generate internal clusters.

    Args:
        context: Asset execution context
        normalized_internal_data: Normalized data for clustering

    Returns:
        Dictionary containing cluster results
    """
    context.log.info("Generating internal clusters")

    # Load clustering configuration from environment-specific config
    config = context.resources.config.load("internal_clustering")
    clustering_config = config.get("clustering", {})

    # Extract clustering parameters from config
    algorithm = clustering_config.get("algorithm", "kmeans")
    n_clusters = clustering_config.get("n_clusters", 5)
    random_state = clustering_config.get("random_state", 42)
    min_cluster_size = clustering_config.get("min_cluster_size", 5)
    cluster_selection_epsilon = clustering_config.get("cluster_selection_epsilon", 0.0)

    # Log configuration
    context.log.info(f"Using algorithm: {algorithm}")
    if algorithm.lower() == "kmeans":
        context.log.info(f"KMeans parameters: n_clusters={n_clusters}, random_state={random_state}")
    elif algorithm.lower() == "hdbscan":
        context.log.info(
            f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, epsilon={cluster_selection_epsilon}"
        )

    # Select features for clustering
    # This should be replaced with your actual feature selection logic
    feature_cols = normalized_internal_data.select(pl.col(pl.Float64)).columns
    features = normalized_internal_data.select(feature_cols).to_numpy()

    # Apply clustering algorithm
    if algorithm.lower() == "kmeans":
        context.log.info(f"Running KMeans with {n_clusters} clusters")
        model = kmeans.KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(features)
    elif algorithm.lower() == "hdbscan":
        context.log.info("Running HDBSCAN")
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
        labels = model.fit_predict(features)
    else:
        error_msg = f"Unsupported clustering algorithm: {algorithm}"
        context.log.error(error_msg)
        context.resources.alerts.alert(message=error_msg, level="ERROR", context={"algorithm": algorithm})
        raise ValueError(error_msg)

    # Add cluster labels to dataframe
    result_df = normalized_internal_data.with_columns(pl.Series("cluster", labels))

    # Compute detailed cluster statistics
    cluster_stats = {}
    cluster_schemas = []

    for cluster_id in result_df["cluster"].unique():
        cluster_rows = result_df.filter(pl.col("cluster") == cluster_id)
        cluster_size = len(cluster_rows)

        # Skip noise points (cluster=-1) in HDBSCAN for statistics
        if algorithm.lower() == "hdbscan" and cluster_id == -1:
            context.log.info(f"Found {cluster_size} noise points")
            continue

        # Compute feature statistics for the cluster
        cluster_features = []
        for col in feature_cols:
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
        if cluster_size > 1 and len(result_df["cluster"].unique()) > 1:
            try:
                # Only calculate for clusters with enough data points
                cluster_features_array = cluster_rows.select(feature_cols).to_numpy()
                if len(cluster_features_array) >= 2:
                    # This is an approximation as we're calculating silhouette per cluster
                    all_but_this_cluster = result_df.filter(pl.col("cluster") != cluster_id)
                    if len(all_but_this_cluster) > 0:
                        other_features = all_but_this_cluster.select(feature_cols).to_numpy()
                        # Use a sample to make computation faster for large clusters
                        sample_size = min(1000, len(cluster_features_array))
                        silhouette_score = float(
                            sk_metrics.silhouette_score(cluster_features_array[:sample_size], [1] * sample_size)
                        )
            except Exception as e:
                context.log.warning(f"Could not calculate silhouette score for cluster {cluster_id}: {e}")

        # Create a validated cluster schema
        cluster_schema = ClusterOutputSchema(
            cluster_id=str(cluster_id), size=cluster_size, features=cluster_features, silhouette_score=silhouette_score
        )

        # Store the validated cluster schema
        cluster_schemas.append(cluster_schema)

        # Also keep the raw stats dictionary for backward compatibility
        cluster_stats[f"cluster_{cluster_id}"] = {
            "count": cluster_size,
            "mean": {col: float(cluster_rows[col].mean()) for col in feature_cols},
            "silhouette_score": silhouette_score,
        }

    # Validate and create the full clustering result
    try:
        clustering_result = ClusteringResult(
            model_version="1.0",
            algorithm=algorithm,
            parameters={
                "n_clusters": n_clusters if algorithm.lower() == "kmeans" else None,
                "min_cluster_size": min_cluster_size if algorithm.lower() == "hdbscan" else None,
                "cluster_selection_epsilon": cluster_selection_epsilon if algorithm.lower() == "hdbscan" else None,
                "random_state": random_state,
            },
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

    except Exception as e:
        error_msg = f"Failed to validate clustering result: {e}"
        context.log.error(error_msg)
        context.resources.alerts.alert(message=error_msg, level="ERROR", context={"error": str(e)})
        raise

    return {
        "clustered_data": result_df,
        "model": model,
        "stats": cluster_stats,
        "result_schema": clustering_result.model_dump(),
    }


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
            feature_cols = clustered_data.select(pl.col(pl.Float64)).columns
            features = clustered_data.select(feature_cols).to_numpy()
            labels = clustered_data["cluster"].to_numpy()

            # Only calculate if we have multiple clusters
            if len(set(labels)) > 1:
                overall_score = float(sk_metrics.silhouette_score(features, labels))
    except Exception as e:
        context.log.warning(f"Could not calculate overall silhouette score: {e}")

    # Check passes if we have valid clusters and no major issues
    passed = valid_clusters > 0 and len(clusters_with_issues) <= 1

    # Return check result with metadata
    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "total_clusters": len(stats),
            "valid_clusters": valid_clusters,
            "problematic_clusters": clusters_with_issues,
            "overall_silhouette_score": overall_score,
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
    internal_clusters: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate internal clustering quality.

    Args:
        context: Asset execution context
        internal_clusters: Internal clustering results

    Returns:
        Dictionary of evaluation metrics
    """
    context.log.info("Evaluating internal clusters")

    clustered_data = internal_clusters["clustered_data"]
    # model variable is not used, so we'll just comment it out
    # model = internal_clusters["model"]

    # Extract features used for clustering
    # This should be replaced with your actual feature extraction logic
    feature_cols = clustered_data.select(pl.col(pl.Float64)).columns
    features = clustered_data.select(feature_cols).to_numpy()

    # Calculate silhouette score if possible
    metrics = {}
    try:
        if len(set(clustered_data["cluster"].unique())) > 1:
            silhouette = sk_metrics.silhouette_score(features, clustered_data["cluster"].to_numpy())
            metrics["silhouette_score"] = silhouette

            # Log the silhouette score with interpretation
            if silhouette > 0.7:
                context.log.info(f"Excellent cluster separation (silhouette={silhouette:.3f})")
            elif silhouette > 0.5:
                context.log.info(f"Good cluster separation (silhouette={silhouette:.3f})")
            elif silhouette > 0.25:
                context.log.info(f"Moderate cluster separation (silhouette={silhouette:.3f})")
            else:
                context.log.warning(f"Poor cluster separation (silhouette={silhouette:.3f})")
    except Exception as e:
        context.log.warning(f"Could not calculate silhouette score: {e}")
        metrics["silhouette_score"] = None

    # Add other evaluation metrics as needed
    # Calinski-Harabasz Index (higher is better)
    try:
        ch_score = sk_metrics.calinski_harabasz_score(features, clustered_data["cluster"].to_numpy())
        metrics["calinski_harabasz_score"] = ch_score
    except Exception as e:
        context.log.warning(f"Could not calculate Calinski-Harabasz score: {e}")

    # Davies-Bouldin Index (lower is better)
    try:
        db_score = sk_metrics.davies_bouldin_score(features, clustered_data["cluster"].to_numpy())
        metrics["davies_bouldin_score"] = db_score
    except Exception as e:
        context.log.warning(f"Could not calculate Davies-Bouldin score: {e}")

    return metrics


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_clusters"],
    compute_kind="internal_clustering",
    group_name="clustering",
    required_resource_keys={"config"},
)
def internal_clustering_output(
    context: dg.AssetExecutionContext,
    internal_clusters: Dict[str, Any],
) -> None:
    """Save internal clustering results.

    Args:
        context: Asset execution context
        internal_clusters: Internal clustering results
    """
    context.log.info("Saving internal clustering results")

    # Load output configuration
    config = context.resources.config.load("internal_clustering")
    output_path = config.get("io", {}).get("output_path", "outputs/internal_clusters")
    output_format = config.get("io", {}).get("format", "parquet")

    clustered_data = internal_clusters["clustered_data"]
    result_schema = internal_clusters.get("result_schema")

    # Save the clustered data
    if hasattr(context.resources, "s3_client") and output_path.startswith("s3://"):
        # Handle S3 output
        s3_client = context.resources.s3_client

        # Parse S3 path
        s3_parts = output_path[5:].split("/", 1)
        bucket = s3_parts[0]

        # Generate key with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        key = f"{s3_parts[1]}/internal_clusters_{timestamp}.{output_format}"

        # Save to temporary file first
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, f"internal_clusters.{output_format}")

            if output_format == "parquet":
                clustered_data.write_parquet(tmp_file)
            elif output_format == "csv":
                clustered_data.write_csv(tmp_file)
            else:
                clustered_data.write_parquet(tmp_file)  # Default to parquet

            # Upload to S3
            s3_client.upload_file(tmp_file, bucket, key)
            context.log.info(f"Uploaded clustering results to s3://{bucket}/{key}")

            # Save schema as JSON if available
            if result_schema:
                import json

                schema_file = os.path.join(tmp_dir, "schema.json")
                with open(schema_file, "w") as f:
                    json.dump(result_schema, f, indent=2)

                schema_key = f"{s3_parts[1]}/internal_clusters_{timestamp}_schema.json"
                s3_client.upload_file(schema_file, bucket, schema_key)
    else:
        # Handle local file output
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output_file = os.path.join(output_dir, f"internal_clusters_{timestamp}.{output_format}")

        if output_format == "parquet":
            clustered_data.write_parquet(output_file)
        elif output_format == "csv":
            clustered_data.write_csv(output_file)
        else:
            clustered_data.write_parquet(output_file)  # Default to parquet

        context.log.info(f"Saved clustering results to {output_file}")

        # Save schema as JSON if available
        if result_schema:
            import json

            schema_file = os.path.join(output_dir, f"internal_clusters_{timestamp}_schema.json")
            with open(schema_file, "w") as f:
                json.dump(result_schema, f, indent=2)

            context.log.info(f"Saved schema to {schema_file}")

    return None
