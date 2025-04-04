"""Assets for merging internal and external clustering results."""

import dagster as dg
import polars as pl

from clustering.utils.helpers import merge_int_ext


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_clustering_output", "external_clustering_output"],
    compute_kind="merging",
    group_name="merging",
)
def merged_clusters(
    context: dg.AssetExecutionContext,
    internal_clustering_output: pl.DataFrame,
    external_clustering_output: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Merge internal and external clustering results.

    Args:
        context: Asset execution context
        internal_clustering_output: Internal clustering results
        external_clustering_output: External clustering results

    Returns:
        Dictionary of merged clustering results by category
    """
    context.log.info("Merging internal and external clustering results")

    # Convert to pandas for the merge_int_ext function
    internal_pd = internal_clustering_output.to_pandas()
    external_pd = external_clustering_output.to_pandas()

    # In the actual implementation, internal_clustering_output should be a dictionary
    # of dataframes by category, but in our simplified version, we'll create a simple
    # dictionary with a single category

    # Create a dictionary with the internal clustering data
    internal_dict = {"Category": internal_pd}

    # For merging with the helper, we need to add a Cluster column to both dataframes
    if "Cluster" not in internal_pd.columns:
        # If there's a cluster column with a different case
        cluster_cols = [col for col in internal_pd.columns if col.lower() == "cluster"]
        if cluster_cols:
            internal_pd.rename(columns={cluster_cols[0]: "Cluster_internal"}, inplace=True)
        else:
            # Create a dummy cluster column
            internal_pd["Cluster_internal"] = 0
    else:
        internal_pd.rename(columns={"Cluster": "Cluster_internal"}, inplace=True)

    if "Cluster" not in external_pd.columns:
        # If there's a cluster column with a different case
        cluster_cols = [col for col in external_pd.columns if col.lower() == "cluster"]
        if cluster_cols:
            external_pd.rename(columns={cluster_cols[0]: "Cluster_external"}, inplace=True)
        else:
            # Create a dummy cluster column
            external_pd["Cluster_external"] = 0
    else:
        external_pd.rename(columns={"Cluster": "Cluster_external"}, inplace=True)

    # Merge the data
    merged_dict_pd = merge_int_ext(internal_dict, external_pd)

    # Convert back to polars
    merged_dict = {k: pl.from_pandas(v) for k, v in merged_dict_pd.items()}

    return merged_dict


@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_clusters"],
    compute_kind="merging",
    group_name="merging",
    required_resource_keys={"output_merged_writer"},
)
def merged_clusters_output(
    context: dg.AssetExecutionContext,
    merged_clusters: dict[str, pl.DataFrame],
) -> None:
    """Save merged clustering results.

    Args:
        context: Asset execution context
        merged_clusters: Merged clustering results
    """
    context.log.info("Saving merged clustering results")

    # Get writer from resources
    merged_output_writer = context.resources.output_merged_writer

    # Check if the writer expects Pandas DataFrame
    if hasattr(merged_output_writer, "requires_pandas") and merged_output_writer.requires_pandas:
        merged_clusters_pd = {k: v.to_pandas() for k, v in merged_clusters.items()}
        merged_output_writer.write(data=merged_clusters_pd)
    else:
        merged_output_writer.write(data=merged_clusters)

    return
