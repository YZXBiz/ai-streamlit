#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for merging and rebalancing cluster labels using centroids.

Place this file at:
  /Users/agrotisnicolas/Clustering_Repo_CVS/utils/merging_centroid_utils.py
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


def load_datasets(internal_file: str, external_file: str):
    """
    Load the internal and external CSVs into pandas DataFrames.

    Parameters
    ----------
    internal_file : str
        Path to the internal CSV file.
    external_file : str
        Path to the external CSV file.

    Returns
    -------
    int_df : pd.DataFrame
        Loaded internal DataFrame.
    ext_df : pd.DataFrame
        Loaded external DataFrame.
    """
    logging.info(f"Loading internal data from: {internal_file}")
    int_df = pd.read_csv(internal_file)
    logging.info(f"Loading external data from: {external_file}")
    ext_df = pd.read_csv(external_file)
    return int_df, ext_df


def extract_granularities(internal_file: str, external_file: str):
    """
    Extract granularity strings from file names.

    Parameters
    ----------
    internal_file : str
        Path to the internal CSV file.
    external_file : str
        Path to the external CSV file.

    Returns
    -------
    internal_granularity : str
        Granularity extracted from the internal file name.
    external_granularity : str
        Granularity extracted from the external file name.
    """
    internal_granularity = os.path.splitext(os.path.basename(internal_file))[0].replace(
        "df_clustered_", ""
    )
    external_granularity = os.path.splitext(os.path.basename(external_file))[0].replace(
        "df_clustered_", ""
    )
    logging.info(f"Internal granularity: {internal_granularity}")
    logging.info(f"External granularity: {external_granularity}")
    return internal_granularity, external_granularity


def merge_cluster_labels(
    int_df: pd.DataFrame, ext_df: pd.DataFrame, internal_granularity: str, external_granularity: str
) -> pd.DataFrame:
    """
    Merge internal and external cluster labels on 'STORE_NBR' and create 'demand_cluster_labels'.

    Parameters
    ----------
    int_df : pd.DataFrame
        Internal DataFrame with columns ["STORE_NBR", "cluster_label"].
    ext_df : pd.DataFrame
        External DataFrame with columns ["STORE_NBR", "cluster_label"].
    internal_granularity : str
        Granularity extracted from internal file name.
    external_granularity : str
        Granularity extracted from external file name.

    Returns
    -------
    df_merged_clusters : pd.DataFrame
        DataFrame containing merged cluster labels, including:
        - store_nbr
        - external_cluster_labels
        - internal_cluster_labels
        - demand_cluster_labels
        - external_granularity
        - internal_granularity
    """
    # Internal
    df_int_clusters = int_df[["STORE_NBR", "cluster_label"]].copy()
    df_int_clusters.rename(
        columns={"STORE_NBR": "store_nbr", "cluster_label": "internal_cluster_labels"}, inplace=True
    )
    df_int_clusters["internal_cluster_labels"] = df_int_clusters["internal_cluster_labels"].astype(
        str
    )

    # External
    df_ext_clusters = ext_df[["STORE_NBR", "cluster_label"]].copy()
    df_ext_clusters.rename(
        columns={"STORE_NBR": "store_nbr", "cluster_label": "external_cluster_labels"}, inplace=True
    )
    df_ext_clusters["external_cluster_labels"] = df_ext_clusters["external_cluster_labels"].astype(
        str
    )

    # Merge on 'store_nbr'
    df_merged_clusters = pd.merge(df_ext_clusters, df_int_clusters, on="store_nbr", how="inner")

    # Create 'demand_cluster_labels'
    df_merged_clusters["demand_cluster_labels"] = (
        df_merged_clusters["external_cluster_labels"]
        + "_"
        + df_merged_clusters["internal_cluster_labels"]
    )

    # Add granularity columns
    df_merged_clusters["external_granularity"] = external_granularity
    df_merged_clusters["internal_granularity"] = internal_granularity

    return df_merged_clusters


def build_feature_matrix(int_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a feature matrix by dropping cluster labels and merging on 'store_nbr'.

    Parameters
    ----------
    int_df : pd.DataFrame
        Internal DataFrame (with 'STORE_NBR' and 'cluster_label' columns).
    ext_df : pd.DataFrame
        External DataFrame (with 'STORE_NBR' and 'cluster_label' columns).

    Returns
    -------
    df_features : pd.DataFrame
        Merged feature DataFrame on 'store_nbr'.
    """
    # Drop cluster_label from internal
    int_feats = int_df.drop(columns=["STORE_NBR", "cluster_label"], errors="ignore").copy()
    int_feats.columns = [f"int_{col}" for col in int_feats.columns]
    int_feats["store_nbr"] = int_df["STORE_NBR"]

    # Drop cluster_label from external
    ext_feats = ext_df.drop(columns=["STORE_NBR", "cluster_label"], errors="ignore").copy()
    ext_feats.columns = [f"ext_{col}" for col in ext_feats.columns]
    ext_feats["store_nbr"] = ext_df["STORE_NBR"]

    # Merge
    df_features = pd.merge(int_feats, ext_feats, on="store_nbr", how="inner")

    return df_features


def merge_and_scale(df_clusters: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cluster labels with feature matrix on 'store_nbr' and standardize numeric columns.

    Parameters
    ----------
    df_clusters : pd.DataFrame
        DataFrame containing cluster labels (must contain 'store_nbr' and 'demand_cluster_labels').
    df_features : pd.DataFrame
        DataFrame containing numeric features merged from internal and external data.

    Returns
    -------
    df_merged : pd.DataFrame
        DataFrame with numeric columns scaled (mean=0, std=1).
        Also includes 'demand_cluster_labels', 'external_cluster_labels', 'internal_cluster_labels',
        'external_granularity', 'internal_granularity'.
    """
    # Merge on 'store_nbr'
    df_merged = pd.merge(df_clusters, df_features, on="store_nbr", how="inner")

    # Columns to exclude from scaling
    exclude_cols = {
        "store_nbr",
        "external_cluster_labels",
        "internal_cluster_labels",
        "demand_cluster_labels",
        "external_granularity",
        "internal_granularity",
    }

    # Identify numeric columns to scale
    numeric_cols = [
        col
        for col in df_merged.columns
        if (col not in exclude_cols and df_merged[col].dtype.kind in ("i", "f"))
    ]

    logging.info(f"Numeric feature columns identified for scaling: {numeric_cols}")

    # Scale numeric columns
    scaler = StandardScaler()
    df_merged[numeric_cols] = scaler.fit_transform(df_merged[numeric_cols])

    return df_merged


def reassign_small_clusters(df_scaled: pd.DataFrame, min_cluster_size=100) -> pd.DataFrame:
    """
    Reassign small clusters to the nearest large cluster based on Euclidean distance to centroids.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        DataFrame with scaled numeric columns and a 'demand_cluster_labels' column.
    min_cluster_size : int, optional
        Clusters with fewer rows than this will be considered 'small' and reassigned.

    Returns
    -------
    df_scaled : pd.DataFrame
        DataFrame with a new column 'rebalanced_demand_cluster_labels' containing the updated clusters.
    """
    # Identify numeric columns (which are scaled)
    exclude_cols = {
        "store_nbr",
        "external_cluster_labels",
        "internal_cluster_labels",
        "demand_cluster_labels",
        "external_granularity",
        "internal_granularity",
    }
    numeric_cols = [
        col
        for col in df_scaled.columns
        if (col not in exclude_cols and df_scaled[col].dtype.kind in ("i", "f"))
    ]

    # Identify cluster sizes
    cluster_sizes = df_scaled["demand_cluster_labels"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index.tolist()
    large_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()

    logging.info(f"Number of small clusters (< {min_cluster_size}): {len(small_clusters)}")
    logging.info(f"Cluster sizes:\n{cluster_sizes}")

    # Compute centroids of large clusters
    centroids_df = df_scaled.groupby("demand_cluster_labels")[numeric_cols].mean()
    centroids_df_large = centroids_df.loc[large_clusters].copy()

    # Create new column for rebalanced clusters
    df_scaled["rebalanced_demand_cluster_labels"] = df_scaled["demand_cluster_labels"].copy()

    # Reassign each store in small clusters to the nearest large cluster
    for small_cl in small_clusters:
        idxs = df_scaled["rebalanced_demand_cluster_labels"] == small_cl
        if idxs.sum() == 0:
            continue

        for row_i in df_scaled[idxs].index:
            store_vec = df_scaled.loc[row_i, numeric_cols].values
            # Compute distance to each large cluster centroid
            distances = []
            for cl_label in centroids_df_large.index:
                centroid_vec = centroids_df_large.loc[cl_label, numeric_cols].values
                dist = np.linalg.norm(store_vec - centroid_vec)
                distances.append((cl_label, dist))
            # Pick the nearest large cluster
            nearest_cluster = min(distances, key=lambda x: x[1])[0]
            df_scaled.at[row_i, "rebalanced_demand_cluster_labels"] = nearest_cluster

    return df_scaled


def build_final_df(
    df_rebalanced: pd.DataFrame, int_df: pd.DataFrame, ext_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the final merged DataFrame, attaching original data from internal & external sources.

    Parameters
    ----------
    df_rebalanced : pd.DataFrame
        DataFrame with rebalanced cluster labels & scaled features.
    int_df : pd.DataFrame
        Original internal DataFrame.
    ext_df : pd.DataFrame
        Original external DataFrame.

    Returns
    -------
    final_df : pd.DataFrame
        The final DataFrame containing all relevant columns and merged data.
    """
    # Rename for merging
    int_final_merge = int_df.drop(columns=["cluster_label"], errors="ignore").copy()
    int_final_merge = int_final_merge.rename(columns={"STORE_NBR": "store_nbr"})

    ext_final_merge = ext_df.drop(columns=["cluster_label"], errors="ignore").copy()
    ext_final_merge = ext_final_merge.rename(columns={"STORE_NBR": "store_nbr"})

    # Subset columns from df_rebalanced for the final output
    final_cols = [
        "store_nbr",
        "external_cluster_labels",
        "internal_cluster_labels",
        "demand_cluster_labels",
        "rebalanced_demand_cluster_labels",
        "external_granularity",
        "internal_granularity",
    ]
    final_df_m1 = df_rebalanced[final_cols].copy()

    # Merge with internal
    final_df_m2 = pd.merge(final_df_m1, int_final_merge, on="store_nbr", how="inner")
    # Merge with external
    final_df_m3 = pd.merge(final_df_m2, ext_final_merge, on="store_nbr", how="inner")

    final_df = final_df_m3.copy()
    return final_df


def save_final_df(
    final_df: pd.DataFrame,
    internal_granularity: str,
    external_granularity: str,
    output_dir: str,
    file_suffix: str = "",
) -> str:
    """
    Save the final DataFrame as a CSV in the specified output directory.

    Parameters
    ----------
    final_df : pd.DataFrame
        Final DataFrame to be saved.
    internal_granularity : str
        Granularity for the internal data.
    external_granularity : str
        Granularity for the external data.
    output_dir : str
        Directory where the output CSV will be written.
    file_suffix : str, optional
        A suffix to append to the file name (e.g., "ALL").

    Returns
    -------
    output_path : str
        Full path to the saved CSV file.
    """
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # If a suffix is provided, prepend an underscore for neatness
    if file_suffix:
        file_suffix = f"_{file_suffix}"

    filename = f"merged_clusters_{internal_granularity}_{external_granularity}{file_suffix}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)

    logging.info(f"Saving final output to: {output_path}")
    final_df.to_csv(output_path, index=False)
    return output_path
