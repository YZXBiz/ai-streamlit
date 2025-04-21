"""Cluster merging assets for the clustering pipeline."""


import dagster as dg
import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances

from clustering.shared.io.readers.pickle_reader import PickleReader


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="merging",
    group_name="merging",
    deps=["internal_save_cluster_assignments", "external_save_cluster_assignments"],
    required_resource_keys={"internal_cluster_assignments_reader", "external_cluster_assignments_reader"},
)
def merged_clusters(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Merge internal and external cluster assignments.

    Loads internal and external cluster assignments from storage,
    merges them based on store numbers, and returns a combined DataFrame
    with cluster assignments from both sources.

    Args:
        context: Dagster asset execution context

    Returns:
        DataFrame with merged cluster assignments
    """
    context.log.info("Starting to merge internal and external clusters")

    # Use the Dagster resources directly
    try:
        internal_clusters = context.resources.internal_cluster_assignments_reader.read()
        context.log.info("Loaded internal clusters from resource")
    except Exception as e:
        raise ValueError(f"Could not read internal cluster assignments: {str(e)}")

    try:
        external_clusters = context.resources.external_cluster_assignments_reader.read()
        context.log.info("Loaded external clusters from resource")
    except Exception as e:
        raise ValueError(f"Could not read external cluster assignments: {str(e)}")

    # If reading returned dictionaries (multiple categories), extract the data
    if isinstance(internal_clusters, dict):
        category = next(iter(internal_clusters.keys()))
        context.log.info(f"Using internal category: {category}")
        internal_clusters = internal_clusters[category]

    if isinstance(external_clusters, dict):
        category = (
            "default" if "default" in external_clusters else next(iter(external_clusters.keys()))
        )
        context.log.info(f"Using external category: {category}")
        external_clusters = external_clusters[category]

    # Verify STORE_NBR is present in both datasets
    if "STORE_NBR" not in internal_clusters.columns:
        raise ValueError("Internal clusters missing STORE_NBR column")

    if "STORE_NBR" not in external_clusters.columns:
        raise ValueError("External clusters missing STORE_NBR column")

    # Join the dataframes on STORE_NBR
    merged = internal_clusters.join(
        external_clusters, on="STORE_NBR", how="inner", suffix="_external"
    )

    # Check if the join resulted in any rows
    if merged.height == 0:
        context.log.error("No common stores found between internal and external data")
        raise ValueError("No common stores found between internal and external data")

    # Remove columns with high null percentage
    total_rows = merged.height
    columns_to_drop = [
        col
        for col in merged.columns
        if col != "STORE_NBR"  # Don't drop the key column
        and merged.select(pl.col(col).is_null().sum()).item() / total_rows > 0.5  # >50% nulls
    ]

    if columns_to_drop:
        context.log.info(f"Dropping {len(columns_to_drop)} columns with high null ratio")
        merged = merged.drop(columns_to_drop)

    context.log.info(f"Successfully merged clusters: {merged.shape} rows and columns")

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
        .agg(pl.len().alias("count"))
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
    }

    # Add min_cluster_size information to the small_clusters DataFrame as metadata
    context.log.info(f"Using min_cluster_size: {min_cluster_size}")

    return result


@dg.asset(
    io_manager_key="io_manager",
    deps=["optimized_merged_clusters"],
    compute_kind="merging",
    group_name="merging",
    required_resource_keys={"internal_model_output", "external_model_output", "job_params"},
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

    # Get minimum cluster size from configuration (now that it's not passed in the dictionary)
    min_cluster_size = (
        context.resources.job_params.min_cluster_size
        if hasattr(context.resources.job_params, "min_cluster_size")
        else 20
    )  # Default to 20 if not specified
    context.log.info(f"Using min_cluster_size: {min_cluster_size}")

    # If no small clusters, return original assignments
    if small_clusters.height == 0:
        context.log.info("No small clusters to reassign")
        return merged_data.select(["STORE_NBR", "merged_cluster"]).with_columns(
            pl.col("merged_cluster").alias("final_cluster")
        )

    # Get the paths from the writer resources
    internal_model_path = context.resources.internal_model_output.path
    external_model_path = context.resources.external_model_output.path

    # Create readers using the same paths as the writers
    internal_model_reader = PickleReader(path=internal_model_path)
    external_model_reader = PickleReader(path=external_model_path)

    # Load model outputs
    try:
        internal_model = internal_model_reader.read()
        external_model = external_model_reader.read()
    except Exception as e:
        context.log.error(f"Error reading model outputs: {str(e)}")
        # Fallback: use original assignments without reassignment
        context.log.warning("Using original cluster assignments without reassignment")
        return merged_data.select(["STORE_NBR", "merged_cluster"]).with_columns(
            pl.col("merged_cluster").alias("final_cluster")
        )

    # If models are dictionaries of models by category, use the first one
    if isinstance(internal_model, dict) and not any(
        k in internal_model for k in ["model", "centroids"]
    ):
        internal_category = next(iter(internal_model.keys()))
        context.log.info(f"Using internal model for category: {internal_category}")
        internal_model = internal_model[internal_category]

    if isinstance(external_model, dict) and not any(
        k in external_model for k in ["model", "centroids"]
    ):
        external_category = (
            "default" if "default" in external_model else next(iter(external_model.keys()))
        )
        context.log.info(f"Using external model for category: {external_category}")
        external_model = external_model[external_category]

    # Extract centroids from models
    if "centroids" not in internal_model or "centroids" not in external_model:
        context.log.error("Could not find centroids in model outputs")
        # Fallback: use original assignments without reassignment
        context.log.warning("Using original cluster assignments without reassignment")
        return merged_data.select(["STORE_NBR", "merged_cluster"]).with_columns(
            pl.col("merged_cluster").alias("final_cluster")
        )

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

            # Explicitly clean up memory
            del large_centroid
            # Force garbage collection periodically
            if len(distances) % 10 == 0:
                import gc

                gc.collect()

        # Assign to nearest large cluster
        if distances:
            nearest_cluster = min(distances, key=distances.get)
            reassignment_map[small_id] = nearest_cluster
        else:
            context.log.warning(f"Could not find any large cluster for reassignment of {small_id}")
            # Keep the original assignment if no reassignment is possible
            reassignment_map[small_id] = small_id

        # Clean up memory after processing each small cluster
        del small_centroid
        del distances
        import gc

        gc.collect()

    # Apply reassignments to create final cluster assignments
    def map_to_final_cluster(cluster_id):
        return reassignment_map.get(cluster_id, cluster_id)

    # Create final cluster assignments
    final_assignments = merged_data.with_columns(
        pl.col("merged_cluster")
        .map_elements(map_to_final_cluster, return_dtype=pl.Utf8)
        .alias("final_cluster")
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
