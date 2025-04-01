"""Utilities for merging different datasets and clustering results."""

from typing import Dict

import polars as pl


def merge_int_ext(
    df_internal: Dict[str, pl.DataFrame],
    df_external: pl.DataFrame,
) -> Dict[str, pl.DataFrame]:
    """Merge internal and external clustering results.

    Combines internal and external clustering results based on store number
    and creates combined cluster identifiers.

    Args:
        df_internal: Dictionary of internal clustering results by category
        df_external: External clustering results dataframe

    Returns:
        Dictionary of merged dataframes by category
    """
    merged_dict = {}

    for category, df_int in df_internal.items():
        # Merge dataframes using Polars join
        merged_df = df_int.join(df_external, on="STORE_NBR", how="inner", suffix="_external")

        # In Polars, we use expressions with string functions for regex replacements
        # Clean up cluster values by removing non-digit characters
        merged_df = merged_df.with_columns(
            [
                pl.col("Cluster").alias("Cluster_internal"),
                pl.col("Cluster_external").cast(pl.Utf8).str.replace_all(r"\D", ""),
            ]
        )

        # Convert internal cluster to string and clean it up as well
        merged_df = merged_df.with_columns(pl.col("Cluster_internal").cast(pl.Utf8).str.replace_all(r"\D", ""))

        # Create combined cluster identifier
        merged_df = merged_df.with_columns(
            (pl.col("Cluster_internal") + "_" + pl.col("Cluster_external")).alias("merged_cluster")
        )

        # Standardize column names to lowercase
        merged_df = merged_df.select([pl.col(col).alias(col.lower()) for col in merged_df.columns])

        # Reorder columns for better readability
        primary_cols = ["store_nbr", "cluster_internal", "cluster_external", "merged_cluster"]
        other_cols = [col for col in merged_df.columns if col not in primary_cols]

        # Select columns in desired order
        merged_dict[category] = merged_df.select(primary_cols + other_cols)

    return merged_dict
