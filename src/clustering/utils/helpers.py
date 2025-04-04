"""Helper functions for the clustering pipeline."""

from typing import Dict, List

import pandas as pd


def merge_dataframes(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes based on common columns.

    Args:
        df_list: List of pandas DataFrames to merge

    Returns:
        Merged DataFrame
    """
    if not df_list:
        return pd.DataFrame()

    if len(df_list) == 1:
        return df_list[0]

    # Start with the first dataframe
    result = df_list[0]

    # Find common columns for joining
    for df in df_list[1:]:
        common_cols = list(set(result.columns) & set(df.columns))
        if common_cols:
            # Use the common columns as join keys
            result = pd.merge(result, df, on=common_cols, how="outer")
        else:
            # If no common columns, use index
            result = pd.merge(result, df, left_index=True, right_index=True, how="outer")

    return result


def merge_int_ext(internal_dict: Dict[str, pd.DataFrame], external_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Merge internal and external clustering results.

    Args:
        internal_dict: Dictionary of internal clustering results by category
        external_df: External clustering results DataFrame

    Returns:
        Dictionary of merged clustering results by category
    """
    result = {}

    # Handle the case where there might be no internal data
    if not internal_dict:
        # Create a dummy category
        result["All"] = external_df.copy()
        return result

    # Merge each internal category with the external data
    for category, internal_df in internal_dict.items():
        # Identify common columns for merging
        common_cols = list(set(internal_df.columns) & set(external_df.columns))

        if common_cols:
            # Merge on common columns
            merged = pd.merge(internal_df, external_df, on=common_cols, how="outer")
        else:
            # If no common columns, try to merge on index
            # First ensure both DataFrames have meaningful indices
            merged = pd.merge(internal_df, external_df, left_index=True, right_index=True, how="outer")

        result[category] = merged

    return result
