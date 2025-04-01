"""Data processing utility functions for clustering operations."""

from typing import Dict, List, Optional

import polars as pl


def clean_ns(
    df_ns: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Clean the need state dataframe.

    Removes duplicates and filters to relevant columns.

    Args:
        df_ns: The dataframe containing need state data
        columns: The columns to keep (defaults to ["PRODUCT_ID", "NEED_STATE"])

    Returns:
        Cleaned need state dataframe
    """
    if columns is None:
        columns = ["PRODUCT_ID", "NEED_STATE"]

    # Polars unique() is equivalent to drop_duplicates in pandas
    return df_ns.unique(subset=columns).select(columns)


def merge_sales_ns(df_sales: pl.DataFrame, df_ns: pl.DataFrame) -> pl.DataFrame:
    """Merge sales data with need state data.

    Merges on SKU_NBR and PRODUCT_ID columns and drops PRODUCT_ID after merge.

    Args:
        df_sales: Sales dataframe with SKU_NBR column
        df_ns: Need state dataframe with PRODUCT_ID and NEED_STATE columns

    Returns:
        Merged dataframe with sales and need state data
    """
    # Polars join with drop=True to drop the join columns from right frame
    return df_sales.join(df_ns, left_on="SKU_NBR", right_on="PRODUCT_ID", how="inner").drop("PRODUCT_ID")


def distribute_sales_evenly(df_merged_sales_ns: pl.DataFrame) -> pl.DataFrame:
    """Distribute sales evenly across need states.

    For cases where a SKU has multiple need states, this distributes the
    sales evenly across all need states for each store.

    Args:
        df_merged_sales_ns: Merged sales and need state dataframe

    Returns:
        Dataframe with sales distributed evenly across need states
    """
    # Get all columns except NEED_STATE for grouping
    group_cols = [col for col in df_merged_sales_ns.columns if col != "NEED_STATE"]

    # Calculate group count using window functions
    with_counts = df_merged_sales_ns.with_columns(pl.col("NEED_STATE").count().over(group_cols).alias("group_count"))

    # Divide TOTAL_SALES by group_count
    result = with_counts.with_columns((pl.col("TOTAL_SALES") / pl.col("group_count")).alias("TOTAL_SALES")).drop(
        "group_count"
    )

    return result


def create_cat_dict(df_NS_inscope: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Create a dictionary of category-specific dataframes.

    Each dataframe contains percentage of sales by need state and store.

    Args:
        df_NS_inscope: Dataframe with need state and sales data

    Returns:
        Dictionary mapping categories to their sales percentage dataframes
    """
    cat_dict = {}

    # Get unique categories
    categories = df_NS_inscope.select("CAT_DSC").unique().to_series().to_list()

    for cat in categories:
        # Filter for this category
        cat_df = df_NS_inscope.filter(pl.col("CAT_DSC") == cat)

        # Store need state sales - group by store and need state
        store_ns_sales = cat_df.group_by(["STORE_NBR", "NEED_STATE"]).agg(
            pl.sum("TOTAL_SALES").alias("STORE_NS_TOTAL_SALES")
        )

        # Store total sales - group by store
        store_sales = cat_df.group_by("STORE_NBR").agg(pl.sum("TOTAL_SALES").alias("STORE_TOTAL_SALES"))

        # Merge store sales with need state sales
        merged = store_ns_sales.join(store_sales, on="STORE_NBR", how="left")

        # Calculate percentage of sales
        with_pct = merged.with_columns(
            (pl.col("STORE_NS_TOTAL_SALES") / pl.col("STORE_TOTAL_SALES") * 100).alias("Pct_of_Sales")
        )

        # Pivot the data
        pivoted = with_pct.pivot(index="STORE_NBR", columns="NEED_STATE", values="Pct_of_Sales").fill_null(0)

        # Rename the columns to "% Sales X"
        rename_map = {col: f"% Sales {col}" for col in pivoted.columns if col != "STORE_NBR"}
        pivoted = pivoted.rename(rename_map)

        # Round the percentage columns to 2 decimal places
        round_exprs = [pl.col(col).round(2) if col != "STORE_NBR" else pl.col(col) for col in pivoted.columns]
        pivoted = pivoted.select(round_exprs)

        cat_dict[cat] = pivoted

    return cat_dict


def merge_dataframes(
    dataframes: List[pl.DataFrame],
    key: str = "STORE_NBR",
    how: str = "outer",
) -> pl.DataFrame:
    """Merge multiple dataframes on a common key.

    Args:
        dataframes: List of dataframes to merge
        key: Column to merge on (default: "STORE_NBR")
        how: Merge type (default: "outer")

    Returns:
        Merged dataframe
    """
    if not dataframes:
        raise ValueError("The list of dataframes is empty")

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        # Get common columns except the key
        common_columns = set(merged_df.columns).intersection(set(df.columns)) - {key}

        # Drop common columns from the current dataframe before merging
        if common_columns:
            merged_df = merged_df.drop(list(common_columns))

        # Join with the next dataframe
        merged_df = merged_df.join(df, on=key, how="inner")

    return merged_df
