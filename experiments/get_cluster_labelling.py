#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import logging
import warnings
from datetime import datetime
from typing import Dict, List

import polars as pl
import numpy as np
from fsutils import run_sf_sql as rp

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Input directory definition
input_directory = "/home/jovyan/fsassortment/store_clustering/merged_content/Merged_Clustering_Output_Run_20250414_1638"

# Load need states data
df_need_states = pl.read_parquet(
    "/home/jovyan/fsassortment/store_clustering/data/need_states_sales_20250414_AM.parquet"
)

df_need_states_mapping = pl.read_csv(
    "/home/jovyan/fsassortment/store_clustering/data/need_states_mapping_20250414_AM.csv"
)

df_need_states_mapping = df_need_states_mapping.select([
    "PRODUCT_ID",
    "NEED_STATE",
    "ATTRIBUTE_1",
    "ATTRIBUTE_2",
    "ATTRIBUTE_3",
    "ATTRIBUTE_4",
    "ATTRIBUTE_5",
    "ATTRIBUTE_6",
    "CDT",
])

# Get SKU category data from Snowflake
conn, _ = rp.get_connection("notebook-xlarge")
query = """
select sku_nbr, category_dsc
from CORE_FSSC.CURATED_PRODUCT.SKU
"""
# Convert pandas DataFrame to Polars
df_sku_cat = pl.from_pandas(pd.read_sql(query, conn))

# Merge SKU category data with need states mapping
df_need_states_mapping = df_need_states_mapping.join(
    df_sku_cat.rename({"SKU_NBR": "PRODUCT_ID", "CATEGORY_DSC": "category"}),
    on="PRODUCT_ID",
    how="left"
)
df_need_states_mapping = df_need_states_mapping.rename({"PRODUCT_ID": "SKU_NBR"})

# Merge need states with need states mapping
df_need_states = df_need_states.join(
    df_need_states_mapping.select([
        "SKU_NBR",
        "NEED_STATE",
        "ATTRIBUTE_1",
        "ATTRIBUTE_2",
        "ATTRIBUTE_3",
        "ATTRIBUTE_4",
        "ATTRIBUTE_5",
        "ATTRIBUTE_6",
        "CDT",
        "category"
    ]),
    on=["SKU_NBR", "NEED_STATE"],
    how="left"
)

# Create a grouped dataset - using polars groupby and agg
df_grouped = df_need_states.group_by([
    "STORE_NBR",
    "NEED_STATE",
    "ATTRIBUTE_1",
    "ATTRIBUTE_2",
    "ATTRIBUTE_3",
    "ATTRIBUTE_4",
    "ATTRIBUTE_5",
    "ATTRIBUTE_6",
    "CDT",
    "CAT_DSC"
], maintain_order=True).agg(
    pl.sum("TOTAL_SALES").alias("TOTAL_SALES")
)

# Prepare output directory with timestamp
current_datetime_str = datetime.now().strftime("%d%m%Y_%H%M")
base_output_directory = "/home/jovyan/fsassortment/store_clustering/classics_content/"
output_directory = os.path.join(
    base_output_directory, f"Classics_Output_Run_{current_datetime_str}"
)
os.makedirs(output_directory, exist_ok=True)

# Columns for clustering data
id_cols = [
    "store_nbr",
    "external_cluster_labels",
    "internal_cluster_labels",
    "demand_cluster_labels",
    "rebalanced_demand_cluster_labels",
    "external_granularity",
    "internal_granularity",
]

# Dictionary to fix categories that have slashes
fix_slashes = {
    "ACNE HSC": "ACNE/HSC",
    "DIET NUTRITION": "DIET/NUTRITION",
}

def build_attr_groupings(df_in: pl.DataFrame, cluster_col: str, attribute_col: str) -> pl.DataFrame:
    """
    Build attribute-specific groupings for analysis.
    
    Args:
        df_in: The aggregated dataframe with TOTAL_SALES_SUM
        cluster_col: Column name for cluster labels
        attribute_col: Column name for attribute to analyze
        
    Returns:
        DataFrame with cluster analysis for the specified attribute
    """
    # Summarize by [cluster_col, CAT_DSC, attribute_col, CDT]
    df_attr = df_in.group_by([cluster_col, "CAT_DSC", attribute_col, "CDT"], maintain_order=True) \
        .agg(pl.sum("TOTAL_SALES_SUM").alias("TOTAL_SALES_SUM"))
    
    # Calculate total CDT total_sales using over expression
    df_attr = df_attr.with_columns(
        pl.col("TOTAL_SALES_SUM").sum().over(["CAT_DSC", "CDT"]).alias("CDT_TOTAL_SALES")
    )
    
    # Calculate total cluster_col total_sales
    df_attr = df_attr.with_columns(
        pl.col("TOTAL_SALES_SUM").sum().over([cluster_col]).alias(f"{cluster_col}_TOTAL_SALES")
    )
    
    # Calculate total_sales across entire dataframe
    total_sum = df_attr.select(pl.sum("TOTAL_SALES_SUM")).item()
    df_attr = df_attr.with_columns(
        pl.lit(total_sum).alias("TOTAL_TOTAL_SALES")
    )
    
    return df_attr

# Process all .csv files from the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        full_path = os.path.join(input_directory, filename)

        # Extract category from the filename
        file_root, _ = os.path.splitext(filename)
        pattern = r"^merged_clusters_(.*?)_\d{8}_\d{4}"
        match = re.match(pattern, file_root)
        if not match:
            print(f"Could not extract category from filename: {filename}")
            continue

        category_underscored = match.group(1)
        category = category_underscored.replace("_", " ")

        # Apply special fixes for slash-based categories
        if category in fix_slashes:
            category = fix_slashes[category]

        print(f"\nProcessing file: {filename}")
        print(f"Detected category: {category}")

        # Read the clustering output
        clustering_output_temp = pl.read_csv(full_path)
        clustering_output = clustering_output_temp.select(id_cols).unique()
        
        # Rename columns for consistency
        clustering_output = clustering_output.rename({
            "store_nbr": "STORE_NBR",
            "external_cluster_labels": "external_cluster_labels",
            "internal_cluster_labels": "internal_cluster_labels",
            "demand_cluster_labels": "demand_cluster_labels",
            "rebalanced_demand_cluster_labels": "rebalanced_demand_cluster_labels",
            "external_granularity": "external_granularity",
            "internal_granularity": "internal_granularity",
        })

        # Filter df_grouped by this category
        df_grouped_CAT = df_grouped.filter(pl.col("CAT_DSC") == category)

        # Merge with clustering_output on STORE_NBR
        df_final_CAT = clustering_output.join(df_grouped_CAT, on="STORE_NBR", how="left")

        # Grouping - Internal
        grouped_internal = df_final_CAT.group_by([
            "internal_cluster_labels",
            "NEED_STATE",
            "CAT_DSC",
            "external_granularity",
            "internal_granularity",
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
            "CDT",
        ], maintain_order=True).agg(
            pl.sum("TOTAL_SALES").alias("TOTAL_SALES_SUM")
        )

        # Grouping - Rebalanced
        grouped_rebalanced = df_final_CAT.group_by([
            "rebalanced_demand_cluster_labels",
            "NEED_STATE",
            "CAT_DSC",
            "external_granularity",
            "internal_granularity",
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
            "CDT",
        ], maintain_order=True).agg(
            pl.sum("TOTAL_SALES").alias("TOTAL_SALES_SUM")
        )

        # Build attribute-specific sheets
        attributes = [
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
        ]
        
        # Build internal attribute sheets
        internal_attr_sheets = {}
        for attr in attributes:
            sheet_df = build_attr_groupings(
                df_in=grouped_internal, 
                cluster_col="internal_cluster_labels", 
                attribute_col=attr
            )
            internal_attr_sheets[f"Grouped Internal {attr}"] = sheet_df

        # Build rebalanced attribute sheets
        rebalanced_attr_sheets = {}
        for attr in attributes:
            sheet_df = build_attr_groupings(
                df_in=grouped_rebalanced,
                cluster_col="rebalanced_demand_cluster_labels",
                attribute_col=attr,
            )
            rebalanced_attr_sheets[f"Grouped Rebalanced {attr}"] = sheet_df

        # Write to Excel in the output directory (convert to pandas for Excel output)
        safe_category_for_filename = category.replace("/", "_")
        excel_filename = f"classics_output_{safe_category_for_filename}_{current_datetime_str}.xlsx"
        excel_full_path = os.path.join(output_directory, excel_filename)

        with pd.ExcelWriter(excel_full_path, engine="xlsxwriter") as writer:
            # Original two sheets
            grouped_internal.to_pandas().to_excel(writer, index=False, sheet_name="Grouped Internal")
            grouped_rebalanced.to_pandas().to_excel(writer, index=False, sheet_name="Grouped Rebalanced")

            # Write attribute-specific sheets (internal)
            for sheet_name, df_sheet in internal_attr_sheets.items():
                df_sheet.to_pandas().to_excel(writer, index=False, sheet_name=sheet_name)

            # Write attribute-specific sheets (rebalanced)
            for sheet_name, df_sheet in rebalanced_attr_sheets.items():
                df_sheet.to_pandas().to_excel(writer, index=False, sheet_name=sheet_name)

        print(f"Excel file created: {excel_full_path}")

        # Print sum checks
        print(
            "Sum of grouped_internal['TOTAL_SALES_SUM'] (rounded):",
            round(grouped_internal.select(pl.sum("TOTAL_SALES_SUM")).item())
        )

        if "TOTAL_SALES" in df_final_CAT.columns:
            print(
                "Sum of df_final_CAT['TOTAL_SALES'] (rounded):",
                round(df_final_CAT.select(pl.sum("TOTAL_SALES")).item())
            )
        else:
            print("df_final_CAT has no 'TOTAL_SALES' column.")

        # Sum for clustering stores
        clustering_stores = clustering_output.select("STORE_NBR").unique().to_series().to_list()
        df_grouped_CAT_for_clustering = df_grouped_CAT.filter(pl.col("STORE_NBR").is_in(clustering_stores))
        
        print(
            "Sum of df_grouped_CAT['TOTAL_SALES'] for clustering stores (rounded):",
            round(df_grouped_CAT_for_clustering.select(pl.sum("TOTAL_SALES")).item())
        )

        # Sum for category & clustering stores
        category_clustering_sum = df_need_states.filter(
            (pl.col("CAT_DSC") == category) & (pl.col("STORE_NBR").is_in(clustering_stores))
        ).select(pl.sum("TOTAL_SALES")).item()
        
        print(
            f"Sum of df_need_states for {category} & clustering stores (rounded):",
            round(category_clustering_sum)
        )

print("\nAll files processed.")