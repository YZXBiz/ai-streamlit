"""
internal_preprocessing.py

Part of the internal data pipeline.
Loads pilot category data, planogram data, and need-state sales data,
then produces a dictionary keyed by category,
where each value is a pivoted DataFrame of % of SALES by NEED_STATE.
"""

import os
import pandas as pd

# Import the internal pipeline configuration
from configs.internal_config import DATA_DIR, PLANOGRAM, PILOT_CATEGORIES, NEED_STATE_PARQUET


def load_internal_data():
    """
    Loads and returns:
      1) df_pilot_cat: DataFrame from pilot_categories.xlsx
      2) df_plano: DataFrame from planogram_dsc.xlsx
      3) df_NS_inscope: DataFrame from need_state_sales_store_correct.parquet
         filtered by the categories in df_pilot_cat['CAT_DSC'].

    Returns:
        df_pilot_cat (pd.DataFrame)
        df_plano (pd.DataFrame)
        df_NS_inscope (pd.DataFrame)
    """
    df_pilot_cat = pd.read_excel(os.path.join(DATA_DIR, PILOT_CATEGORIES))
    df_plano = pd.read_excel(os.path.join(DATA_DIR, PLANOGRAM))

    # Identify in-scope categories from pilot file
    in_scope_cat = df_pilot_cat["CAT_DSC"].unique().tolist()

    # Load the need-state sales data
    df_need_states = pd.read_parquet(os.path.join(DATA_DIR, NEED_STATE_PARQUET))
    # df_need_states = pd.read_csv(os.path.join(DATA_DIR, NEED_STATE_PARQUET))

    # Filter the need-state data by in-scope categories
    df_NS_inscope = df_need_states[df_need_states["CAT_DSC"].isin(in_scope_cat)].copy()

    return df_pilot_cat, df_plano, df_NS_inscope


def create_cat_dict(df_NS_inscope):
    """
    Takes df_NS_inscope and creates a dictionary keyed by 'CAT_DSC',
    where each value is a pivoted DataFrame showing % of Sales by NEED_STATE.

    Returns:
        cat_dict (dict): {category_str: pivoted_pdDataFrame}
    """
    categories = df_NS_inscope["CAT_DSC"].unique()
    cat_dict = {}

    for cat in categories:
        # 1) Subset the data to the current category
        cat_df = df_NS_inscope[df_NS_inscope["CAT_DSC"] == cat].copy()

        # 2) Group by (STORE_NBR, NEED_STATE)
        grouped = cat_df.groupby(["STORE_NBR", "NEED_STATE"], as_index=False)["TOTAL_SALES"].sum()

        # 3) Compute total store sales
        store_totals = grouped.groupby("STORE_NBR", as_index=False)["TOTAL_SALES"].sum()
        store_totals.rename(columns={"TOTAL_SALES": "STORE_TOTAL_SALES"}, inplace=True)

        # 4) Merge to get store totals on each row
        merged = grouped.merge(store_totals, on="STORE_NBR", how="left")

        # 5) Compute percent of total sales
        merged["Pct_of_Sales"] = merged["TOTAL_SALES"] / merged["STORE_TOTAL_SALES"] * 100.0

        # 6) Pivot so each NEED_STATE becomes a column
        pivoted = merged.pivot(index="STORE_NBR", columns="NEED_STATE", values="Pct_of_Sales")

        # 7) Rename columns
        pivoted.columns = [f"% Sales {col}" for col in pivoted.columns]
        pivoted.reset_index(inplace=True)

        # 8) Fill missing values with 0
        pivoted.fillna(0, inplace=True)

        # 9) Round numeric columns except STORE_NBR
        cols_to_round = pivoted.columns.difference(["STORE_NBR"])
        pivoted[cols_to_round] = pivoted[cols_to_round].round(2)

        # 10) Store this category's DataFrame in the dictionary
        cat_dict[cat] = pivoted

    return cat_dict


def sanitize_name(name: str) -> str:
    """
    Replace or remove filesystem-unsafe characters from the name
    so we can safely use it in filenames. For instance, '/' becomes '_'.
    """
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("?", "")
        .replace("*", "")
        .replace(":", "")
        # Add more if needed (e.g., <, >, |, etc.)
    )
