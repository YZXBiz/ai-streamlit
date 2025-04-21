#!/usr/bin/env python
"""Test script for cluster labeling analytics - full version."""

import os
import pickle
import pandas as pd
from datetime import datetime

# Set up paths
output_path = "data/internal"
input_directory = "data/merging"

# Create timestamped output directory
current_datetime_str = datetime.now().strftime("%d%m%Y_%H%M")
output_directory = os.path.join(
    output_path, f"Classics_Output_Run_{current_datetime_str}"
)
os.makedirs(output_directory, exist_ok=True)
print(f"Created output directory: {output_directory}")

# Load merged cluster assignments
try:
    with open(os.path.join(input_directory, "merged_cluster_assignments.pkl"), 'rb') as f:
        merged_clusters_dict = pickle.load(f)
    print(f"Loaded merged cluster assignments with {len(merged_clusters_dict)} categories")
    print(f"Categories: {list(merged_clusters_dict.keys())}")
except Exception as e:
    print(f"Error loading merged cluster assignments: {str(e)}")
    raise

# Load need states sales data
try:
    df_need_states = pd.read_csv(
        os.path.join(output_path, "ns_sales.csv"),
    )
    print(f"Loaded need states sales data with {df_need_states.shape[0]} rows")
    print(f"Need states columns: {df_need_states.columns.tolist()}")
except Exception as e:
    print(f"Error loading need states sales data: {str(e)}")
    raise

# Load need states mapping data
try:
    df_need_states_mapping = pd.read_csv(
        os.path.join(output_path, "ns_map.csv")
    )
    print(f"Loaded need states mapping data with {df_need_states_mapping.shape[0]} rows")
    print(f"Need states mapping columns: {df_need_states_mapping.columns.tolist()}")
except Exception as e:
    print(f"Error loading need states mapping data: {str(e)}")
    raise

# Prepare need states mapping
try:
    needed_columns = [
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
    
    # Check if all needed columns exist
    missing_columns = [col for col in needed_columns if col not in df_need_states_mapping.columns]
    if missing_columns:
        print(f"Warning: Missing columns in df_need_states_mapping: {missing_columns}")
        # Add placeholder columns for any missing ones
        for col in missing_columns:
            df_need_states_mapping[col] = "UNKNOWN"
    
    df_need_states_mapping = df_need_states_mapping[needed_columns]
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
except Exception as e:
    print(f"Error preparing need states mapping: {str(e)}")
    raise

# Merge need states data with mapping
try:
    rows_before = df_need_states.shape[0]
    
    # Check and rename columns if needed
    if "SKU_NBR" in df_need_states.columns and "PRODUCT_ID" not in df_need_states.columns:
        df_need_states = df_need_states.rename(columns={"SKU_NBR": "PRODUCT_ID"})
    
    # Merge on PRODUCT_ID
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
    rows_after = df_need_states.shape[0]
    print(f"Merged need states data: {rows_before} rows before, {rows_after} rows after merge")
except Exception as e:
    print(f"Error merging need states data: {str(e)}")
    raise

# Create grouped dataset
try:
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
    
    print(f"Created grouped dataset with {df_grouped.shape[0]} rows")
except Exception as e:
    print(f"Error creating grouped dataset: {str(e)}")
    raise

# Dictionary to fix categories that have slashes, etc.
fix_slashes = {
    "ACNE HSC": "ACNE/HSC",
    "DIET NUTRITION": "DIET/NUTRITION",
}

# ID columns for cluster assignments
id_cols = [
    "STORE_NBR",
    "external_cluster",
    "internal_cluster",
    "merged_cluster",
    "rebalanced_cluster",
]

# Helper function to build attribute-specific sheets
def build_attr_groupings(df_in, cluster_col, attribute_col):
    """Build attribute-specific dataframes with sales aggregations."""
    try:
        # Summarize by [cluster_col, CAT_DSC, attribute_col, CDT]
        df_attr = (
            df_in.groupby([cluster_col, "CAT_DSC", attribute_col, "CDT"], dropna=False)
            .agg({"TOTAL_SALES_SUM": "sum"})
            .reset_index()
        )

        # Total CDT total_sales
        df_attr["CDT_TOTAL_SALES"] = df_attr.groupby(["CAT_DSC", "CDT"])[
            "TOTAL_SALES_SUM"
        ].transform("sum")

        # Total cluster_col total_sales
        df_attr[f"{cluster_col}_TOTAL_SALES"] = df_attr.groupby([cluster_col])[
            "TOTAL_SALES_SUM"
        ].transform("sum")

        # Total total_sales
        total_sum = df_attr["TOTAL_SALES_SUM"].sum()
        df_attr["TOTAL_TOTAL_SALES"] = total_sum

        return df_attr
    except Exception as e:
        print(f"Error building attribute groupings: {str(e)}")
        return None

# Process each category
for category, clustering_output_temp in merged_clusters_dict.items():
    try:
        print(f"\nProcessing category: {category}")
        
        # Convert to pandas if needed
        if hasattr(clustering_output_temp, 'to_pandas'):
            clustering_output_temp = clustering_output_temp.to_pandas()
        
        # Check if clustering_output_temp has all required columns
        missing_cols = [col for col in id_cols if col not in clustering_output_temp.columns]
        if missing_cols:
            print(f"Warning: Missing columns in clustering output: {missing_cols}")
            print(f"Available columns: {clustering_output_temp.columns.tolist()}")
        
        # Keep only the ID columns that exist
        available_id_cols = [col for col in id_cols if col in clustering_output_temp.columns]
        if not available_id_cols:
            print(f"No ID columns available in clustering output for category {category}. Skipping.")
            continue
        
        clustering_output = clustering_output_temp[available_id_cols].drop_duplicates().copy()
        
        # Apply special fixes for slash-based categories if needed
        fixed_category = fix_slashes.get(category, category)
        df_grouped_cat = df_grouped[df_grouped["CAT_DSC"] == fixed_category]
        
        if df_grouped_cat.empty:
            print(f"Warning: No data for category {fixed_category} in df_grouped")
            print(f"Available categories: {df_grouped['CAT_DSC'].unique().tolist()}")
            continue
        
        # Merge with clustering_output on STORE_NBR
        df_final_cat = clustering_output.merge(df_grouped_cat, on="STORE_NBR", how="left")
        print(f"Merged output has {df_final_cat.shape[0]} rows")
        
        # Check if we have the necessary columns for grouping
        if "internal_cluster" not in df_final_cat.columns or "rebalanced_cluster" not in df_final_cat.columns:
            missing = []
            if "internal_cluster" not in df_final_cat.columns:
                missing.append("internal_cluster")
            if "rebalanced_cluster" not in df_final_cat.columns:
                missing.append("rebalanced_cluster")
            print(f"Missing required columns for grouping: {missing}")
            continue
            
        # Grouping - Internal
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
        
        # Grouping - Rebalanced
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
        
        # Build attribute sheets for internal clusters
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
            if sheet_df is not None:
                internal_attr_sheets[f"Grouped Internal {attr}"] = sheet_df
        
        # Build attribute sheets for rebalanced clusters
        rebalanced_attr_sheets = {}
        for attr in attributes:
            sheet_df = build_attr_groupings(
                df_in=grouped_rebalanced,
                cluster_col="rebalanced_cluster",
                attribute_col=attr,
            )
            if sheet_df is not None:
                rebalanced_attr_sheets[f"Grouped Rebalanced {attr}"] = sheet_df
        
        # Write to Excel in the output directory
        safe_category_for_filename = fixed_category.replace("/", "_")
        excel_filename = f"classics_output_{safe_category_for_filename}_{current_datetime_str}.xlsx"
        excel_full_path = os.path.join(output_directory, excel_filename)
        
        print(f"Writing Excel file: {excel_filename}")
        
        with pd.ExcelWriter(excel_full_path, engine="xlsxwriter") as writer:
            # Original two sheets:
            grouped_internal.to_excel(writer, index=False, sheet_name="Grouped Internal")
            grouped_rebalanced.to_excel(writer, index=False, sheet_name="Grouped Rebalanced")
            
            # Write the attribute-specific sheets (internal)
            for sheet_name, df_sheet in internal_attr_sheets.items():
                df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)
            
            # Write the attribute-specific sheets (rebalanced)
            for sheet_name, df_sheet in rebalanced_attr_sheets.items():
                df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)
        
        print(f"Excel file created: {excel_full_path}")
        
        # Print sum checks
        print(f"Sum checks for category {fixed_category}:")
        print(
            f"Sum of grouped_internal['TOTAL_SALES_SUM'] (rounded): {round(grouped_internal['TOTAL_SALES_SUM'].sum())}"
        )
        
        # Sum of df_final_cat['TOTAL_SALES']
        if "TOTAL_SALES" in df_final_cat.columns:
            print(
                f"Sum of df_final_cat['TOTAL_SALES'] (rounded): {round(df_final_cat['TOTAL_SALES'].sum())}"
            )
        
        # Sum of df_grouped_cat['TOTAL_SALES'] for the clustering stores
        clustering_stores = clustering_output["STORE_NBR"].unique().tolist()
        df_grouped_cat_for_clustering = df_grouped_cat[
            df_grouped_cat["STORE_NBR"].isin(clustering_stores)
        ]
        print(
            f"Sum of df_grouped_cat['TOTAL_SALES'] for clustering stores (rounded): "
            f"{round(df_grouped_cat_for_clustering['TOTAL_SALES'].sum())}"
        )
        
        # Sum of df_need_states for this category & clustering stores
        category_clustering_sum = df_need_states[
            (df_need_states["CAT_DSC"] == fixed_category)
            & (df_need_states["STORE_NBR"].isin(clustering_stores))
        ]["TOTAL_SALES"].sum()
        print(
            f"Sum of df_need_states for {fixed_category} & clustering stores (rounded): "
            f"{round(category_clustering_sum)}"
        )
    except Exception as e:
        print(f"Error processing category {category}: {str(e)}")

print("\nAll categories processed. Test completed!") 