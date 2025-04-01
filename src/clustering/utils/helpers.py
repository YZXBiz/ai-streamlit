# %% IMPORTS
import os
import dask.dataframe as pd
import re
from typing import List


# %% FUNCTIONS
def clean_ns(
    df_ns: pd.DataFrame,
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Cleans the need state dataframe by removing duplicates and filtering relevant columns.

    This function ensures that only unique combinations of the specified columns are retained.

    Args:
        df_ns (pd.DataFrame): The dataframe containing the need state data.
        columns (List[str]): The list of columns to retain and clean. Defaults to ["PRODUCT_ID", "NEED_STATE"].

    Returns:
        pd.DataFrame: A cleaned dataframe with only the specified columns, containing no duplicates or null values.
    """
    if columns is None:
        columns = ["PRODUCT_ID", "NEED_STATE"]

    df_ns_de_dup = df_ns.drop_duplicates(subset=columns)
    df_ns_filter = df_ns_de_dup[columns]  # Retain only the specified columns
    return df_ns_filter


def merge_sales_ns(df_sales: pd.DataFrame, df_ns: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the sales dataframe with the need state dataframe on the "SKU_NBR" and "PRODUCT_ID" columns.

    This function merges the sales data (`df_sales`) with the need state data (`df_ns`) by matching the
    "SKU_NBR" column in the sales dataframe with the "PRODUCT_ID" column in the need state dataframe.
    It ensures that the resulting dataframe contains sales data along with the corresponding need state
    information. The "PRODUCT_ID" column is dropped after the merge.

    Args:
        df_sales (pd.DataFrame): The dataframe containing the sales data. Must include a "SKU_NBR" column.
        df_ns (pd.DataFrame): The dataframe containing the need state data. Must include "PRODUCT_ID" and
                              "NEED_STATE" columns.

    Returns:
        pd.DataFrame: The merged dataframe, containing sales data with the corresponding need state
                      information, and with the "PRODUCT_ID" column removed.
    """
    df_merged_sales_ns = df_sales.merge(
        df_ns,
        how="inner",
        left_on=["SKU_NBR"],
        right_on=["PRODUCT_ID"],
    ).drop("PRODUCT_ID", axis=1)

    return df_merged_sales_ns


def distribute_sales_evenly(df_merged_sales_ns: pd.DataFrame) -> pd.DataFrame:
    """
    Distributes the sales evenly across all need states for each store.

    There are some duplicated need states for some SKUs in some stores.
    This function distributes the sales evenly across all need states for each store.
    Otherwise, you will run into issues like this:
        - SKU_NBR: 123, NEED_STATE: "A", TOTAL_SALES: 100
        - SKU_NBR: 123, NEED_STATE: "B", TOTAL_SALES: 100
    Here, the TOTAL_SALES for SKU_NBR: 123 is 200, but the actual TOTAL_SALES for NEED_STATE: "A" and "B" should be 100.
    To fix this, we need to divide the TOTAL_SALES by the number of need states for each store.
    Then, we can get the corrected TOTAL_SALES which is 50 for both need states:
        - SKU_NBR: 123, NEED_STATE: "A", TOTAL_SALES: 50
        - SKU_NBR: 123, NEED_STATE: "B", TOTAL_SALES: 50

    Args:
        df_merged_sales_ns (pd.DataFrame): The dataframe containing the merged sales and need state data.

    Returns:
        pd.DataFrame: The dataframe with the sales distributed evenly across all need states for each store.
    """
    group_cols = [col for col in df_merged_sales_ns.columns if col != "NEED_STATE"]

    return df_merged_sales_ns.assign(
        group_count=lambda df: df.groupby(group_cols)["NEED_STATE"].transform("count"),
        TOTAL_SALES=lambda df: df["TOTAL_SALES"] / df["group_count"],
    ).drop(columns=["group_count"])


def create_cat_dict(df_NS_inscope: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Creates a dictionary of dataframes, where each dataframe contains the percentage of sales for each need state in each store.

    Args:
        df_NS_inscope (pd.DataFrame): The dataframe containing the need state data.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of dataframes, where each dataframe contains the percentage of sales for each need state in each store.
    """
    cat_dict = {}
    for cat in df_NS_inscope["CAT_DSC"].unique():
        # * Stores' need state sales
        store_ns_sales = (
            df_NS_inscope[df_NS_inscope["CAT_DSC"] == cat]
            .groupby(["STORE_NBR", "NEED_STATE"], as_index=False)
            .agg({"TOTAL_SALES": "sum"})
            .rename(columns={"TOTAL_SALES": "STORE_NS_TOTAL_SALES"})
        )

        # * Stores' total sales
        store_sales = (
            df_NS_inscope[df_NS_inscope["CAT_DSC"] == cat]
            .groupby("STORE_NBR", as_index=False)
            .agg({"TOTAL_SALES": "sum"})
            .rename(columns={"TOTAL_SALES": "STORE_TOTAL_SALES"})
        )

        # * Merge store need state sales with store total sales
        pivoted = (
            store_ns_sales.merge(
                store_sales,
                on="STORE_NBR",
                how="left",
            )
            .assign(Pct_of_Sales=lambda df: (df["STORE_NS_TOTAL_SALES"] / df["STORE_TOTAL_SALES"]) * 100.0)
            .pivot(index="STORE_NBR", columns="NEED_STATE", values="Pct_of_Sales")
            .fillna(0)
            .reset_index()
            .rename(columns=lambda col: f"% Sales {col}" if col != "STORE_NBR" else col)
            .pipe(lambda df: df.round({col: 2 for col in df.columns if col != "STORE_NBR"}))
            .rename_axis(None, axis=1)
        )

        cat_dict[cat] = pivoted

    return cat_dict


def merge_dataframes(
    dataframes: list[pd.DataFrame],
    key: str = "STORE_NBR",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Merge a list of dataframes on a common key.

    Parameters:
        dataframes (List[pd.DataFrame]): List of dataframes to merge.
        key (str): The column name to merge on. Defaults to "STORE_NBR".
        how (str): Type of merge to be performed. Defaults to "inner".

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    if not dataframes:
        raise ValueError("The list of dataframes is empty")

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        # Drop duplicate columns before merging
        common_columns = merged_df.columns.intersection(df.columns).tolist()
        common_columns.remove(key)  # Keep the key column
        merged_df = merged_df.drop(columns=common_columns).merge(df, how=how, on=key)

    return merged_df


def merge_int_ext(
    df_internal: dict[str, pd.DataFrame],
    df_external: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Merges the internal and external features on the "STORE_NBR" column.

    Parameters:
        df_internal (Dict[str, pd.DataFrame]): The dictionary containing the internal features.
        df_external (pd.DataFrame): The dataframe containing the external features.

    Returns:
        Dict[str, pd.DataFrame]: The dictionary containing the merged dataframes.
    """
    merged_dict = dict()
    for category, df_int in df_internal.items():
        merged_df = df_int.merge(
            df_external,
            on="STORE_NBR",
            suffixes=("_internal", "_external"),
        )
        # Remove non-numerical values from Cluster_internal and Cluster_external using regex
        merged_df["Cluster_internal"] = merged_df["Cluster_internal"].astype(str).apply(lambda x: re.sub(r"\D", "", x))
        merged_df["Cluster_external"] = merged_df["Cluster_external"].astype(str).apply(lambda x: re.sub(r"\D", "", x))

        merged_df["merged_cluster"] = merged_df["Cluster_internal"] + "_" + merged_df["Cluster_external"]

        merged_dict[category] = merged_df

        # lower all the columns to lower case
        merged_dict[category].columns = map(str.lower, merged_dict[category].columns)

        # move store_nbr, cluster_internal, cluster_external and demand_cluster to the front
        merged_dict[category] = merged_dict[category][
            ["store_nbr", "cluster_internal", "cluster_external", "merged_cluster"]
            + [
                col
                for col in merged_dict[category].columns
                if col not in ["store_nbr", "cluster_internal", "cluster_external", "merged_cluster"]
            ]
        ]

    return merged_dict


def create_classic_output(
    df_need_states_sales: pd.DataFrame,
    df_need_states_mapping: pd.DataFrame,
    df_sku: pd.DataFrame,
) -> pd.DataFrame:
    df_merge_sku_mapping = df_need_states_mapping.merge(df_sku, on="PRODUCT_ID", how="inner")


def handle_dvc_lineage(
    folder_path: str, commit_message: str, remote_name: str = "clusteringremote", push_to_remote: bool = False
) -> None:
    """
    Handles DVC lineage for a given folder. It automatically adds the folder to DVC, Git, and pushes the data to a remote.
    - dvc add folder_path
    - git add folder_path.dvc .gitignore
    - git commit -m commit_message
    - dvc push -r remote_name

    Parameters:
        folder_path (str): The path to the folder or file to be tracked by DVC.
        remote_name (str): The name of the DVC remote to push the data to.
        commit_message (str): The commit message for Git.
    """
    os.system(f"dvc add {folder_path}")
    os.system(f"git add {folder_path}.dvc .gitignore")
    os.system(f"git commit -m '{commit_message}'")
    if push_to_remote:
        os.system(f"dvc push -r {remote_name}")
