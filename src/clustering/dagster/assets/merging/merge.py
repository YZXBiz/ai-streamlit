"""Cluster merging assets for the clustering pipeline."""

import dagster as dg
import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="merging",
    group_name="merging",
    deps=["internal_save_cluster_assignments", "external_save_cluster_assignments"],
    required_resource_keys={"internal_cluster_assignments", "external_cluster_assignments"},
)
def merged_clusters(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load and merge internal and external cluster assignments.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing merged cluster assignments
    """
    context.log.info("Loading internal and external cluster assignments")

    # Load internal and external cluster assignments
    internal_clusters = context.resources.internal_cluster_assignments.read()
    external_clusters = context.resources.external_cluster_assignments.read()

    # Ensure both dataframes have STORE_NBR column
    if "STORE_NBR" not in internal_clusters.columns:
        raise ValueError("Internal clusters missing STORE_NBR column")
    if "STORE_NBR" not in external_clusters.columns:
        raise ValueError("External clusters missing STORE_NBR column")

    # Join the dataframes on STORE_NBR
    context.log.info("Joining internal and external clusters")
    merged = internal_clusters.join(
        external_clusters, on="STORE_NBR", how="inner", suffix="_external"
    )

    # Find the cluster columns
    internal_cluster_col = [
        col for col in internal_clusters.columns if "cluster" in col.lower() and col != "STORE_NBR"
    ][0]
    external_cluster_col = [
        col for col in external_clusters.columns if "cluster" in col.lower() and col != "STORE_NBR"
    ][0]

    if not internal_cluster_col or not external_cluster_col:
        raise ValueError("Could not identify cluster columns in the data")

    # Create merged cluster identifier
    merged = merged.with_columns(
        (
            pl.col(internal_cluster_col).cast(pl.Utf8)
            + "_"
            + pl.col(external_cluster_col).cast(pl.Utf8)
        ).alias("merged_cluster")
    )

    context.log.info(
        f"Created {merged.select(pl.col('merged_cluster').n_unique())} merged clusters"
    )

    return merged


@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_clusters"],
    compute_kind="merging",
    group_name="merging",
)
def merged_cluster_assignments(
    context: dg.AssetExecutionContext,
    merged_clusters: pl.DataFrame,
) -> dict[str, dict]:
    """Create a mapping of merged cluster assignments with counts.

    Args:
        context: Asset execution context
        merged_clusters: Merged cluster assignments

    Returns:
        Dictionary with merged cluster mappings and counts
    """
    context.log.info("Calculating merged cluster statistics")

    # Count occurrences of each merged cluster
    cluster_counts = (
        merged_clusters.group_by("merged_cluster")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )

    # Convert to dictionary for easier access
    cluster_map = {
        "clusters": cluster_counts.to_dict(as_series=False),
        "store_mappings": merged_clusters.select(["STORE_NBR", "merged_cluster"]).to_dict(
            as_series=False
        ),
    }

    return cluster_map


@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_clusters", "merged_cluster_assignments"],
    compute_kind="merging",
    group_name="merging",
    required_resource_keys={"job_params"},
)
def optimized_merged_clusters(
    context: dg.AssetExecutionContext,
    merged_clusters: pl.DataFrame,
    merged_cluster_assignments: dict[str, dict],
) -> dict[str, pl.DataFrame]:
    """Identify small clusters that need reassignment.

    Args:
        context: Asset execution context
        merged_clusters: Merged cluster data
        merged_cluster_assignments: Cluster assignment mappings

    Returns:
        Dictionary containing small and large clusters
    """
    # Get minimum cluster size from configuration
    min_cluster_size = (
        context.resources.job_params.min_cluster_size
        if hasattr(context.resources.job_params, "min_cluster_size")
        else 20
    )  # Default to 20 if not specified

    context.log.info(f"Identifying clusters smaller than {min_cluster_size}")

    # Extract cluster counts
    cluster_counts = pl.DataFrame(merged_cluster_assignments["clusters"])

    # Separate small and large clusters
    small_clusters = cluster_counts.filter(pl.col("count") < min_cluster_size)
    large_clusters = cluster_counts.filter(pl.col("count") >= min_cluster_size)

    context.log.info(
        f"Found {small_clusters.height} small clusters and {large_clusters.height} large clusters"
    )

    # Create a comprehensive dictionary with all data needed for reassignment
    result = {
        "small_clusters": small_clusters,
        "large_clusters": large_clusters,
        "merged_data": merged_clusters,
        "min_cluster_size": min_cluster_size,
    }

    return result


@dg.asset(
    io_manager_key="io_manager",
    deps=["optimized_merged_clusters"],
    compute_kind="merging",
    group_name="merging",
    required_resource_keys={"internal_model_output", "external_model_output"},
)
def cluster_reassignment(
    context: dg.AssetExecutionContext,
    optimized_merged_clusters: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Reassign small clusters to their nearest large cluster using centroids.

    Args:
        context: Asset execution context
        optimized_merged_clusters: Dictionary with cluster data

    Returns:
        DataFrame with final cluster assignments
    """
    context.log.info("Reassigning small clusters to nearest large clusters")

    # Extract data from input
    small_clusters = optimized_merged_clusters["small_clusters"]
    large_clusters = optimized_merged_clusters["large_clusters"]
    merged_data = optimized_merged_clusters["merged_data"]

    # If no small clusters, return original assignments
    if small_clusters.height == 0:
        context.log.info("No small clusters to reassign")
        return merged_data.select(["STORE_NBR", "merged_cluster"]).with_columns(
            pl.col("merged_cluster").alias("final_cluster")
        )

    # Get cluster models to access centroids
    internal_model = context.resources.internal_model_output.read()
    external_model = context.resources.external_model_output.read()

    # Extract centroids from models
    if "centroids" not in internal_model or "centroids" not in external_model:
        raise ValueError("Could not find centroids in model outputs")

    internal_centroids = internal_model["centroids"]
    external_centroids = external_model["centroids"]

    # Create a mapping of original clusters to their centroids
    all_centroids = {}
    for i_cluster, i_centroid in internal_centroids.items():
        for e_cluster, e_centroid in external_centroids.items():
            merged_id = f"{i_cluster}_{e_cluster}"
            # Combine centroids (assuming they are numpy arrays)
            all_centroids[merged_id] = np.concatenate([i_centroid, e_centroid])

    # Get the small and large cluster IDs
    small_cluster_ids = small_clusters.select("merged_cluster").to_series().to_list()
    large_cluster_ids = large_clusters.select("merged_cluster").to_series().to_list()

    # Create mapping of small clusters to nearest large cluster
    reassignment_map = {}
    for small_id in small_cluster_ids:
        # Get centroid of small cluster
        small_centroid = all_centroids.get(small_id)
        if small_centroid is None:
            context.log.warning(f"No centroid found for small cluster {small_id}")
            continue

        # Calculate distances to all large clusters
        distances = {}
        for large_id in large_cluster_ids:
            large_centroid = all_centroids.get(large_id)
            if large_centroid is None:
                continue

            # Calculate Euclidean distance between centroids
            distance = pairwise_distances(
                small_centroid.reshape(1, -1), large_centroid.reshape(1, -1), metric="euclidean"
            )[0][0]
            distances[large_id] = distance

        # Assign to nearest large cluster
        if distances:
            nearest_cluster = min(distances, key=distances.get)
            reassignment_map[small_id] = nearest_cluster
        else:
            context.log.warning(f"Could not find any large cluster for reassignment of {small_id}")
            # Keep the original assignment if no reassignment is possible
            reassignment_map[small_id] = small_id

    # Apply reassignments to create final cluster assignments
    def map_to_final_cluster(cluster_id):
        return reassignment_map.get(cluster_id, cluster_id)

    # Create final cluster assignments
    final_assignments = merged_data.with_columns(
        pl.col("merged_cluster").map_elements(map_to_final_cluster).alias("final_cluster")
    ).select(["STORE_NBR", "merged_cluster", "final_cluster"])

    # Log reassignment stats
    reassigned_count = merged_data.filter(pl.col("merged_cluster").is_in(small_cluster_ids)).height

    context.log.info(f"Reassigned {reassigned_count} stores from small clusters")

    return final_assignments


@dg.asset(
    name="save_merged_cluster_assignments",
    description="Saves final merged cluster assignments to storage",
    group_name="merging",
    compute_kind="merged_cluster_assignment",
    deps=["cluster_reassignment"],
    required_resource_keys={"merged_cluster_assignments"},
)
def save_merged_cluster_assignments(
    context: dg.AssetExecutionContext,
    cluster_reassignment: pl.DataFrame,
) -> None:
    """Save final merged cluster assignments to persistent storage.

    Uses the configured output resource to save the merged cluster assignments
    for later use in analysis or reporting.

    Args:
        context: Dagster asset execution context
        cluster_reassignment: DataFrame with final cluster assignments
    """
    context.log.info("Saving merged cluster assignments to storage")

    # Use the configured output resource
    assignments_output = context.resources.merged_cluster_assignments

    # Save the assignments
    assignments_output.write(cluster_reassignment)

    context.log.info("Successfully saved merged cluster assignments")
