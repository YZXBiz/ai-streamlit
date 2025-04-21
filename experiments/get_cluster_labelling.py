#!/usr/bin/env python
"""Script to process cluster assignments and generate analytics output files.

This script reads need states sales data and merged cluster assignments,
processes the data by category, and outputs Excel files with various analytics
breakdowns by cluster labels.
"""

import logging
import os
import pickle
from datetime import datetime

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Pandas Display Settings for Full Data Views
pd.set_option("display.max_rows", 100)  # Show all rows
pd.set_option("display.max_columns", 1000)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Prevent column width truncation

# ----------------------------------------------------------------
# 1. Read data at the beginning
# ----------------------------------------------------------------
# Point to the final merge output directory
input_directory = "/workspaces/clustering-dagster/data/merging"

# Read need states sales data
df_need_states = pd.read_csv(
    "/workspaces/clustering-dagster/data/internal/ns_sales.csv",
)

# Read need states mapping data
df_need_states_mapping = pd.read_csv(
    "/workspaces/clustering-dagster/data/internal/ns_map.csv"
)

# Rename columns if needed to match the expected format
df_need_states_mapping = df_need_states_mapping[
    [
        "PRODUCT_ID", 
        "NEED_STATE",
        "ATTRIBUTE_1",
        "ATTRIBUTE_2",
        "ATTRIBUTE_3",
        "ATTRIBUTE_4",
        "ATTRIBUTE_5",
        "ATTRIBUTE_6",
        "CDT",
        "CATEGORY",
    ]
]
df_need_states_mapping.columns = [
    "PRODUCT_ID",
    "NEED_STATE",
    "ATTRIBUTE_1",
    "ATTRIBUTE_2",
    "ATTRIBUTE_3",
    "ATTRIBUTE_4",
    "ATTRIBUTE_5",
    "ATTRIBUTE_6",
    "CDT",
    "category",
]

# ----------------------------------------------------------------
# 2. Merge df_need_states and df_need_states_mapping
# ----------------------------------------------------------------
rows_before = df_need_states.shape[0]
# Rename SKU_NBR in df_need_states to match PRODUCT_ID in df_need_states_mapping
df_need_states = df_need_states.rename(columns={"SKU_NBR": "PRODUCT_ID"})

# Now merge only on PRODUCT_ID
df_need_states = df_need_states.merge(
    df_need_states_mapping[
        [
            "PRODUCT_ID",
            "NEED_STATE",
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
            "CDT",
            "category",
        ]
    ],
    on=["PRODUCT_ID"],
    how="left",
)

# ----------------------------------------------------------------
# 3. Create a grouped dataset (df_grouped) with dropna=False
# ----------------------------------------------------------------
df_grouped = df_need_states.groupby(
    [
        "STORE_NBR",
        "NEED_STATE",
        "ATTRIBUTE_1",
        "ATTRIBUTE_2",
        "ATTRIBUTE_3",
        "ATTRIBUTE_4",
        "ATTRIBUTE_5",
        "ATTRIBUTE_6",
        "CDT",
        "CAT_DSC",
    ],
    dropna=False,
    as_index=False,
)["TOTAL_SALES"].sum()

# ----------------------------------------------------------------
# 4. Prepare output directory with timestamp
# ----------------------------------------------------------------
current_datetime_str = datetime.now().strftime("%d%m%Y_%H%M")
base_output_directory = "/workspaces/clustering-dagster/data/internal/"
output_directory = os.path.join(
    base_output_directory, f"Classics_Output_Run_{current_datetime_str}"
)
os.makedirs(output_directory, exist_ok=True)

# ----------------------------------------------------------------
# 5. Process all merged cluster assignments from the pickle file
# ----------------------------------------------------------------
# Updated ID columns to match your actual column names
id_cols = [
    "STORE_NBR",
    "external_cluster",
    "internal_cluster",
    "merged_cluster",
    "rebalanced_cluster",
]

# Dictionary to fix categories that have slashes, etc.
fix_slashes = {
    "ACNE HSC": "ACNE/HSC",
    "DIET NUTRITION": "DIET/NUTRITION",
    # add more if needed
}

# Load the merged cluster assignments
with open(os.path.join(input_directory, "merged_cluster_assignments.pkl"), 'rb') as f:
    merged_clusters_dict = pickle.load(f)

for category, clustering_output_temp in merged_clusters_dict.items():
    
    print(f"\nProcessing category: {category}")
    
    # --------------------------------------------------------
    # b) Prepare the clustering output
    # --------------------------------------------------------
    # Keep only the ID columns, drop duplicates
    clustering_output = clustering_output_temp[id_cols].drop_duplicates().copy()

    # --------------------------------------------------------
    # c) Filter df_grouped by this category
    # --------------------------------------------------------
    # Apply special fixes for slash-based categories if needed
    fixed_category = fix_slashes.get(category, category)
    df_grouped_cat = df_grouped[df_grouped["CAT_DSC"] == fixed_category]

    # --------------------------------------------------------
    # d) Merge with clustering_output on STORE_NBR
    # --------------------------------------------------------
    df_final_cat = clustering_output.merge(df_grouped_cat, on="STORE_NBR", how="left")

    # --------------------------------------------------------
    # e) Grouping - Internal
    # --------------------------------------------------------
    grouped_internal = (
        df_final_cat.groupby(
            [
                "internal_cluster",
                "NEED_STATE",
                "CAT_DSC",
                "ATTRIBUTE_1",
                "ATTRIBUTE_2",
                "ATTRIBUTE_3",
                "ATTRIBUTE_4",
                "ATTRIBUTE_5",
                "ATTRIBUTE_6",
                "CDT",
            ],
            dropna=False,
        )
        .agg(TOTAL_SALES_SUM=("TOTAL_SALES", "sum"))
        .reset_index()
    )

    # --------------------------------------------------------
    # f) Grouping - Rebalanced
    # --------------------------------------------------------
    grouped_rebalanced = (
        df_final_cat.groupby(
            [
                "rebalanced_cluster",
                "NEED_STATE",
                "CAT_DSC",
                "ATTRIBUTE_1",
                "ATTRIBUTE_2",
                "ATTRIBUTE_3",
                "ATTRIBUTE_4",
                "ATTRIBUTE_5",
                "ATTRIBUTE_6",
                "CDT",
            ],
            dropna=False,
        )
        .agg(TOTAL_SALES_SUM=("TOTAL_SALES", "sum"))
        .reset_index()
    )

    # --------------------------------------------------------
    # 5.1: Helper function to build "attribute-specific" sheets
    # --------------------------------------------------------
    def build_attr_groupings(df_in, cluster_col, attribute_col):
        """Build attribute-specific dataframes with sales aggregations.
        
        df_in        : The aggregated dataframe (e.g. grouped_internal or grouped_rebalanced),
                       which has TOTAL_SALES_SUM already.
        cluster_col  : 'internal_cluster' or 'rebalanced_cluster'
        attribute_col: 'ATTRIBUTE_1', 'ATTRIBUTE_2', etc.

        Returns a new dataframe with:
          groupby([cluster_col, 'CAT_DSC', attribute_col, 'CDT'], dropna=False).sum()
          plus extra columns:
            1) total CDT total_sales
            2) total cluster_col total_sales
            3) total total_sales across entire df
        """
        # Summarize by [cluster_col, CAT_DSC, attribute_col, CDT]
        df_attr = (
            df_in.groupby([cluster_col, "CAT_DSC", attribute_col, "CDT"], dropna=False)
            .agg({"TOTAL_SALES_SUM": "sum"})
            .reset_index()
        )

        # total CDT total_sales
        df_attr["CDT_TOTAL_SALES"] = df_attr.groupby(["CAT_DSC", "CDT"])[
            "TOTAL_SALES_SUM"
        ].transform("sum")

        # total cluster_col total_sales
        df_attr[f"{cluster_col}_TOTAL_SALES"] = df_attr.groupby([cluster_col])[
            "TOTAL_SALES_SUM"
        ].transform("sum")

        # total total_sales
        total_sum = df_attr["TOTAL_SALES_SUM"].sum()
        df_attr["TOTAL_TOTAL_SALES"] = total_sum

        return df_attr

    # --------------------------------------------------------
    # 5.2: Build all the extra sheets for grouped_internal
    # --------------------------------------------------------
    attributes = [
        "ATTRIBUTE_1",
        "ATTRIBUTE_2",
        "ATTRIBUTE_3",
        "ATTRIBUTE_4",
        "ATTRIBUTE_5",
        "ATTRIBUTE_6",
    ]
    internal_attr_sheets = {}
    for attr in attributes:
        sheet_df = build_attr_groupings(
            df_in=grouped_internal, cluster_col="internal_cluster", attribute_col=attr
        )
        internal_attr_sheets[f"Grouped Internal {attr}"] = sheet_df

    # --------------------------------------------------------
    # 5.3: Build all the extra sheets for grouped_rebalanced
    # --------------------------------------------------------
    rebalanced_attr_sheets = {}
    for attr in attributes:
        sheet_df = build_attr_groupings(
            df_in=grouped_rebalanced,
            cluster_col="rebalanced_cluster",
            attribute_col=attr,
        )
        rebalanced_attr_sheets[f"Grouped Rebalanced {attr}"] = sheet_df

    # --------------------------------------------------------
    # g) Write to Excel in the new output directory
    # --------------------------------------------------------
    # Replace any slashes in 'category' so it's safe as a file name
    safe_category_for_filename = fixed_category.replace("/", "_")
    excel_filename = f"classics_output_{safe_category_for_filename}_{current_datetime_str}.xlsx"
    excel_full_path = os.path.join(output_directory, excel_filename)

    with pd.ExcelWriter(excel_full_path, engine="xlsxwriter") as writer:
        # Original two sheets:
        grouped_internal.to_excel(writer, index=False, sheet_name="Grouped Internal")
        grouped_rebalanced.to_excel(writer, index=False, sheet_name="Grouped Rebalanced")

        # Write the new attribute-specific sheets (internal)
        for sheet_name, df_sheet in internal_attr_sheets.items():
            df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)

        # Write the new attribute-specific sheets (rebalanced)
        for sheet_name, df_sheet in rebalanced_attr_sheets.items():
            df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Excel file created: {excel_full_path}")

    # --------------------------------------------------------
    # h) Print sum checks (rounded to the nearest whole number)
    # --------------------------------------------------------
    # 1) Sum of grouped_internal['TOTAL_SALES_SUM']
    print(
        "Sum of grouped_internal['TOTAL_SALES_SUM'] (rounded):",
        round(grouped_internal["TOTAL_SALES_SUM"].sum()),
    )

    # 2) Sum of df_final_cat['TOTAL_SALES']
    if "TOTAL_SALES" in df_final_cat.columns:
        print(
            "Sum of df_final_cat['TOTAL_SALES'] (rounded):",
            round(df_final_cat["TOTAL_SALES"].sum()),
        )
    else:
        print("df_final_cat has no 'TOTAL_SALES' column.")

    # 3) Sum of df_grouped_cat['TOTAL_SALES'] for the clustering stores
    clustering_stores = clustering_output["STORE_NBR"].unique().tolist()
    df_grouped_cat_for_clustering = df_grouped_cat[
        df_grouped_cat["STORE_NBR"].isin(clustering_stores)
    ]
    print(
        "Sum of df_grouped_cat['TOTAL_SALES'] for clustering stores (rounded):",
        round(df_grouped_cat_for_clustering["TOTAL_SALES"].sum()),
    )

    # 4) Sum of df_need_states for this category & clustering stores
    category_clustering_sum = df_need_states[
        (df_need_states["CAT_DSC"] == fixed_category)
        & (df_need_states["STORE_NBR"].isin(clustering_stores))
    ]["TOTAL_SALES"].sum()
    print(
        f"Sum of df_need_states for {fixed_category} & clustering stores (rounded):",
        round(category_clustering_sum),
    )

print("\nAll categories processed.")

