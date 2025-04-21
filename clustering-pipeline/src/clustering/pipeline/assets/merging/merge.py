"""Cluster merging assets for the clustering pipeline."""


import dagster as dg
import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances
import os

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
        
    # Log cluster columns for debugging
    context.log.info(f"Internal columns available: {internal_clusters.columns}")
    context.log.info(f"External columns available: {external_clusters.columns}")
    
    # Check if internal clusters has all the features we need
    # If not, try to find and load internal features dataset
    if len(internal_clusters.columns) <= 3:  # Only has STORE_NBR, Cluster, and maybe category
        context.log.info("Internal clusters dataset has limited columns, checking for internal features")
        try:
            # Try to load internal features from a few possible locations
            possible_paths = [
                "data/internal/internal_sales_by_category.pkl",
                "data/internal/sales_by_category.pkl",
            ]
            
            # Initialize a dictionary to hold features by store number
            store_features = {}
            
            # Loop through all paths and combine features
            for path in possible_paths:
                if os.path.exists(path):
                    context.log.info(f"Found internal features at {path}")
                    try:
                        import pickle
                        with open(path, 'rb') as f:
                            internal_features = pickle.load(f)
                            
                        # If it's a dictionary, we need to handle each category
                        if isinstance(internal_features, dict):
                            # Loop through all categories
                            for category_key, category_df in internal_features.items():
                                context.log.info(f"Processing category: {category_key}")
                                
                                # Convert to polars if needed
                                if not isinstance(category_df, pl.DataFrame):
                                    category_df = pl.from_pandas(category_df)
                                
                                # For each store, add the features
                                for row in category_df.iter_rows(named=True):
                                    store_nbr = row["STORE_NBR"]
                                    if store_nbr not in store_features:
                                        store_features[store_nbr] = {}
                                    
                                    # Add each feature from this category
                                    for col, val in row.items():
                                        if col != "STORE_NBR":
                                            store_features[store_nbr][col] = val
                        else:
                            # It's a single DataFrame
                            # Convert to polars if needed
                            if not isinstance(internal_features, pl.DataFrame):
                                internal_features = pl.from_pandas(internal_features)
                            
                            # For each store, add the features
                            for row in internal_features.iter_rows(named=True):
                                store_nbr = row["STORE_NBR"]
                                if store_nbr not in store_features:
                                    store_features[store_nbr] = {}
                                
                                # Add each feature from this row
                                for col, val in row.items():
                                    if col != "STORE_NBR":
                                        store_features[store_nbr][col] = val
                    except Exception as e:
                        context.log.warning(f"Error loading internal features from {path}: {str(e)}")
            
            # Now create a DataFrame from the combined features
            if store_features:
                # Convert the dictionary to lists of records
                records = []
                for store_nbr, features in store_features.items():
                    record = {"STORE_NBR": store_nbr}
                    record.update(features)
                    records.append(record)
                
                # Create a polars DataFrame
                combined_features = pl.DataFrame(records)
                context.log.info(f"Created combined features DataFrame with {combined_features.shape[1]} columns")
                
                # Merge with internal clusters
                internal_clusters = internal_clusters.join(
                    combined_features, 
                    on="STORE_NBR", 
                    how="left",
                    suffix="_features"
                )
                context.log.info(f"Enhanced internal clusters with features, now has {internal_clusters.shape[1]} columns")
                context.log.info(f"Internal feature columns: {[col for col in internal_clusters.columns if col not in ['STORE_NBR', 'Cluster', 'category']]}")
        except Exception as e:
            context.log.warning(f"Failed to load additional internal features: {str(e)}")
    
    # Verify or standardize Cluster columns
    if "Cluster" not in internal_clusters.columns:
        if "cluster" in internal_clusters.columns:
            internal_clusters = internal_clusters.rename({"cluster": "internal_cluster"})
        else:
            # Handle case where there's no cluster column
            context.log.warning("No cluster column found in internal data, creating default")
            internal_clusters = internal_clusters.with_columns(pl.lit(0).alias("internal_cluster"))
    else:
        # Rename Cluster to internal_cluster
        internal_clusters = internal_clusters.rename({"Cluster": "internal_cluster"})
    
    # Clean internal cluster values - remove any "Cluster" prefix from values
    internal_clusters = internal_clusters.with_columns(
        pl.col("internal_cluster").str.replace("Cluster ", "").str.replace("cluster ", "").alias("internal_cluster")
    )
            
    if "Cluster" not in external_clusters.columns:
        if "cluster" in external_clusters.columns:
            external_clusters = external_clusters.rename({"cluster": "external_cluster"})
        else:
            # Handle case where there's no cluster column
            context.log.warning("No cluster column found in external data, creating default")
            external_clusters = external_clusters.with_columns(pl.lit(0).alias("external_cluster"))
    else:
        # Rename Cluster to external_cluster
        external_clusters = external_clusters.rename({"Cluster": "external_cluster"})
    
    # Clean external cluster values - remove any "Cluster" prefix from values
    external_clusters = external_clusters.with_columns(
        pl.col("external_cluster").str.replace("Cluster ", "").str.replace("cluster ", "").alias("external_cluster")
    )

    # Join the dataframes on STORE_NBR
    merged = internal_clusters.join(
        external_clusters, on="STORE_NBR", how="inner", suffix="_external"
    )

    # Check if the join resulted in any rows
    if merged.height == 0:
        context.log.error("No common stores found between internal and external data")
        raise ValueError("No common stores found between internal and external data")

    # Log columns before cleanup
    context.log.info(f"Columns after join: {merged.columns}")
    
    # KEEP ALL COLUMNS - do not drop any columns
    """
    # Handle any *_external columns where we want to prefer the internal version
    external_cols_to_drop = []
    for col in merged.columns:
        if col.endswith("_external") and col != "external_cluster":
            base_col = col.replace("_external", "")
            if base_col in merged.columns:
                external_cols_to_drop.append(col)
                context.log.info(f"Will drop {col} in favor of {base_col}")
    
    if external_cols_to_drop:
        merged = merged.drop(external_cols_to_drop)
        context.log.info(f"Dropped {len(external_cols_to_drop)} duplicate external columns")
    """

    # Remove columns with high null percentage
    """
    total_rows = merged.height
    columns_to_drop = [
        col
        for col in merged.columns
        if col != "STORE_NBR"  # Don't drop the key column
        and col != "internal_cluster"  # Don't drop the internal cluster column
        and col != "external_cluster"  # Don't drop the external cluster column
        and merged.select(pl.col(col).is_null().sum()).item() / total_rows > 0.5  # >50% nulls
    ]

    if columns_to_drop:
        context.log.info(f"Dropping {len(columns_to_drop)} columns with high null ratio")
        merged = merged.drop(columns_to_drop)
    """

    # Create merged_cluster column explicitly
    # Convert cluster columns to strings and concatenate with underscore
    try:
        merged = merged.with_columns(
            pl.concat_str([
                pl.col("internal_cluster").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("external_cluster").cast(pl.Utf8)
            ]).alias("merged_cluster")
        )
        context.log.info("Created merged_cluster column from internal and external clusters")
        
        # Log sample of merged values for verification
        sample_merged = merged.select(["internal_cluster", "external_cluster", "merged_cluster"]).head(5)
        context.log.info(f"Sample of merged values: {sample_merged}")
    except Exception as e:
        context.log.warning(f"Error creating merged_cluster column: {str(e)}")
        context.log.info("Creating alternative merged_cluster column")
        # Fallback to creating sequential cluster IDs
        merged = merged.with_row_count("row_id")
        merged = merged.with_columns(
            pl.concat_str([
                pl.lit("merged_"),
                pl.col("row_id").cast(pl.Utf8)
            ]).alias("merged_cluster")
        )
        merged = merged.drop("row_id")

    context.log.info(f"Successfully merged clusters: {merged.shape} rows and columns")
    context.log.info(f"Final merged columns: {merged.columns}")

    # Reorganize columns to ensure key columns are at the front
    priority_columns = ["STORE_NBR"]
    
    # Add category if it exists
    if "category" in merged.columns:
        priority_columns.append("category")
    
    # Add internal and external cluster columns
    if "internal_cluster" in merged.columns:
        priority_columns.append("internal_cluster")
    
    if "external_cluster" in merged.columns:
        priority_columns.append("external_cluster")
    
    # Add merged_cluster
    if "merged_cluster" in merged.columns:
        priority_columns.append("merged_cluster")
    
    # Add remaining columns
    remaining_columns = [col for col in merged.columns if col not in priority_columns]
    
    # Reorder columns
    merged = merged.select(priority_columns + remaining_columns)
    context.log.info(f"Reorganized columns: {merged.columns}")

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

    # Check available columns and log for debugging
    context.log.info(f"Available columns in merged_clusters: {merged_clusters.columns}")

    # Create a merged_cluster column if it doesn't exist
    if "merged_cluster" not in merged_clusters.columns:
        context.log.info("Creating merged_cluster column from cluster columns")
        
        # Determine the column names based on what's available
        internal_cluster_col = "internal_cluster" if "internal_cluster" in merged_clusters.columns else None
        external_cluster_col = "external_cluster" if "external_cluster" in merged_clusters.columns else None
        
        if internal_cluster_col and external_cluster_col:
            # Create a new column that combines internal and external clusters
            merged_clusters = merged_clusters.with_columns(
                pl.concat_str([
                    pl.col(internal_cluster_col).cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col(external_cluster_col).cast(pl.Utf8)
                ]).alias("merged_cluster")
            )
            context.log.info(f"Created merged_cluster column from {internal_cluster_col} and {external_cluster_col}")
        elif internal_cluster_col:
            # Just use the internal_cluster if that's all we have
            merged_clusters = merged_clusters.with_columns(
                pl.col(internal_cluster_col).cast(pl.Utf8).alias("merged_cluster")
            )
            context.log.info(f"Created merged_cluster column from internal {internal_cluster_col} only")
        else:
            # Fallback to creating sequential IDs if no cluster columns are found
            context.log.warning("No cluster columns found, creating sequential cluster IDs")
            merged_clusters = merged_clusters.with_row_count("merged_cluster")
            # Convert row numbers to string format for consistency
            merged_clusters = merged_clusters.with_columns(
                pl.col("merged_cluster").cast(pl.Utf8).alias("merged_cluster")
            )
    
    # Check if category column exists for organizing by category
    has_category = "category" in merged_clusters.columns
    context.log.info(f"Dataset has category column: {has_category}")
    
    # Create the cluster map with categories as keys
    cluster_map = {}
    
    if has_category:
        # Get unique categories
        categories = merged_clusters.select("category").unique().to_series().to_list()
        context.log.info(f"Found {len(categories)} unique categories")
        
        # Process each category
        for category in categories:
            # Filter data for this category
            category_data = merged_clusters.filter(pl.col("category") == category)
            
            # Count occurrences of each merged cluster in this category
            try:
                category_counts = (
                    category_data.group_by("merged_cluster")
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                )
                
                # Store category data in the map
                cluster_map[str(category)] = {
                    "clusters": category_counts.to_dict(as_series=False),
                    "store_mappings": category_data.select(["STORE_NBR", "merged_cluster"]).to_dict(as_series=False)
                }
                context.log.info(f"Added cluster data for category: {category}")
            except Exception as e:
                context.log.warning(f"Error processing category {category}: {str(e)}")
                # Add empty entry
                cluster_map[str(category)] = {
                    "clusters": {"merged_cluster": [], "count": []},
                    "store_mappings": {"STORE_NBR": [], "merged_cluster": []}
                }
    else:
        # No category column, use "all" as the key
        try:
            cluster_counts = (
                merged_clusters.group_by("merged_cluster")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            
            cluster_map["all"] = {
                "clusters": cluster_counts.to_dict(as_series=False),
                "store_mappings": merged_clusters.select(["STORE_NBR", "merged_cluster"]).to_dict(as_series=False)
            }
            context.log.info("Added cluster data with 'all' as category key")
        except Exception as e:
            context.log.error(f"Error calculating cluster statistics: {str(e)}")
            # Provide debug information
            context.log.info(f"DataFrame shape: {merged_clusters.shape}")
            context.log.info(f"Column dtypes: {merged_clusters.dtypes}")
            
            # Return a minimal dictionary with empty mappings
            cluster_map["all"] = {
                "clusters": {"merged_cluster": [], "count": []},
                "store_mappings": {"STORE_NBR": [], "merged_cluster": []}
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

    # Initialize results dictionary
    result = {
        "small_clusters": None,
        "large_clusters": None,
        "merged_data": merged_clusters,
    }

    # Process by category if available
    if len(merged_cluster_assignments) > 0:
        # Since we've restructured by category, we'll aggregate across all categories
        all_small_clusters_data = []
        all_large_clusters_data = []
        
        # Process each category
        for category, category_data in merged_cluster_assignments.items():
            context.log.info(f"Processing category: {category}")
            
            if not category_data["clusters"]["merged_cluster"]:
                context.log.warning(f"No clusters found for category: {category}")
                continue
                
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
            
            context.log.info(
                f"Category {category}: {small_clusters.height} small clusters and {large_clusters.height} large clusters"
            )
        
        # Combine results from all categories
        if all_small_clusters_data:
            result["small_clusters"] = pl.concat(all_small_clusters_data)
        else:
            result["small_clusters"] = pl.DataFrame({"merged_cluster": [], "count": []})
            
        if all_large_clusters_data:
            result["large_clusters"] = pl.concat(all_large_clusters_data)
        else:
            result["large_clusters"] = pl.DataFrame({"merged_cluster": [], "count": []})
    else:
        # Handle the case where no categorization exists
        context.log.warning("No cluster assignments found. Creating empty cluster sets.")
        result["small_clusters"] = pl.DataFrame({"merged_cluster": [], "count": []})
        result["large_clusters"] = pl.DataFrame({"merged_cluster": [], "count": []})

    context.log.info(
        f"Total: {result['small_clusters'].height} small clusters and {result['large_clusters'].height} large clusters"
    )

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
) -> dict[str, pl.DataFrame]:
    """Reassign small clusters to their nearest large cluster using centroids.

    Args:
        context: Asset execution context
        optimized_merged_clusters: Dictionary with cluster data

    Returns:
        Dictionary of DataFrames with final cluster assignments organized by category
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

    # If no small clusters, create final assignments without reassignment
    if small_clusters.height == 0:
        context.log.info("No small clusters to reassign")
        final_assignments_df = merged_data.with_columns(
            pl.col("merged_cluster").alias("rebalanced_cluster")
        )
        
        # Organize by category
        return organize_by_category(context, final_assignments_df)

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
        final_assignments_df = merged_data.with_columns(
            pl.col("merged_cluster").alias("rebalanced_cluster")
        )
        
        # Organize by category
        return organize_by_category(context, final_assignments_df)

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
        final_assignments_df = merged_data.with_columns(
            pl.col("merged_cluster").alias("rebalanced_cluster")
        )
        
        # Organize by category
        return organize_by_category(context, final_assignments_df)

    internal_centroids = internal_model["centroids"]
    external_centroids = external_model["centroids"]

    # Create a mapping of original clusters to their centroids
    all_centroids = {}
    for i_cluster, i_centroid in internal_centroids.items():
        # Clean internal_cluster key if needed
        i_cluster_clean = str(i_cluster).replace("Cluster ", "").replace("cluster ", "")
        
        for e_cluster, e_centroid in external_centroids.items():
            # Clean external_cluster key if needed
            e_cluster_clean = str(e_cluster).replace("Cluster ", "").replace("cluster ", "")
            
            merged_id = f"{i_cluster_clean}_{e_cluster_clean}"
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

    # Create final cluster assignments - preserve all columns and add rebalanced_cluster
    final_assignments_df = merged_data.with_columns(
        pl.col("merged_cluster")
        .map_elements(map_to_final_cluster, return_dtype=pl.Utf8)
        .alias("rebalanced_cluster")
    )

    # Log reassignment stats
    reassigned_count = merged_data.filter(pl.col("merged_cluster").is_in(small_cluster_ids)).height

    context.log.info(f"Reassigned {reassigned_count} stores from small clusters")
    context.log.info(f"Final assignments dataframe has {final_assignments_df.shape[0]} rows and {final_assignments_df.shape[1]} columns")
    context.log.info(f"Columns in final assignments: {final_assignments_df.columns}")

    # Reorganize columns to ensure key columns are at the front
    priority_columns = ["STORE_NBR"]
    
    # Add category if it exists
    if "category" in final_assignments_df.columns:
        priority_columns.append("category")
    
    # Add internal and external cluster columns
    if "internal_cluster" in final_assignments_df.columns:
        priority_columns.append("internal_cluster")
    
    if "external_cluster" in final_assignments_df.columns:
        priority_columns.append("external_cluster")
    
    # Add merged_cluster followed immediately by rebalanced_cluster
    if "merged_cluster" in final_assignments_df.columns:
        priority_columns.append("merged_cluster")
    
    if "rebalanced_cluster" in final_assignments_df.columns:
        priority_columns.append("rebalanced_cluster")
    
    # Add remaining columns
    remaining_columns = [col for col in final_assignments_df.columns if col not in priority_columns]
    
    # Reorder columns
    final_assignments_df = final_assignments_df.select(priority_columns + remaining_columns)
    context.log.info(f"Reorganized columns: {final_assignments_df.columns}")
    
    # Make an additional explicit reordering to place rebalanced_cluster right after merged_cluster
    all_columns = final_assignments_df.columns
    # Find where rebalanced_cluster is
    if "rebalanced_cluster" in all_columns and "merged_cluster" in all_columns:
        rebalanced_index = all_columns.index("rebalanced_cluster")
        merged_index = all_columns.index("merged_cluster")
        
        # If rebalanced_cluster is not right after merged_cluster, move it
        if rebalanced_index != merged_index + 1:
            all_columns.remove("rebalanced_cluster")
            all_columns.insert(merged_index + 1, "rebalanced_cluster")
            final_assignments_df = final_assignments_df.select(all_columns)
            context.log.info("Explicitly moved rebalanced_cluster column to be right after merged_cluster")

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
    # Before splitting by category, ensure rebalanced_cluster is right after merged_cluster
    if "rebalanced_cluster" in dataframe.columns and "merged_cluster" in dataframe.columns:
        # Get all column names
        all_columns = list(dataframe.columns)
        
        # Find the indices
        merged_idx = all_columns.index("merged_cluster")
        rebalanced_idx = all_columns.index("rebalanced_cluster")
        
        # If rebalanced_cluster is not immediately after merged_cluster, rearrange
        if rebalanced_idx != merged_idx + 1:
            # Remove rebalanced_cluster from its current position
            all_columns.remove("rebalanced_cluster")
            
            # Insert it right after merged_cluster
            all_columns.insert(merged_idx + 1, "rebalanced_cluster")
            
            # Reorder the dataframe
            dataframe = dataframe.select(all_columns)
            context.log.info("Rearranged columns to place rebalanced_cluster right after merged_cluster")
    
    # Check if category column exists
    has_category = "category" in dataframe.columns
    
    result = {}
    
    if has_category:
        # Get unique categories
        categories = dataframe.select("category").unique().to_series().to_list()
        context.log.info(f"Organizing results by {len(categories)} categories")
        
        # Create a DataFrame for each category
        for category in categories:
            category_df = dataframe.filter(pl.col("category") == category)
            result[str(category)] = category_df
            context.log.info(f"Category {category}: {category_df.shape[0]} stores")
    else:
        # No category, use "all" as the key
        context.log.info("No category column found, using 'all' as the category key")
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

    Uses the configured output resource to save the merged cluster assignments
    for later use in analysis or reporting.

    Args:
        context: Dagster asset execution context
        cluster_reassignment: Dictionary of DataFrames with final cluster assignments by category
    """
    context.log.info("Saving merged cluster assignments to storage")

    # Use the configured output resource
    assignments_output = context.resources.merged_cluster_assignments

    # Save the assignments
    assignments_output.write(cluster_reassignment)

    context.log.info("Successfully saved merged cluster assignments")
