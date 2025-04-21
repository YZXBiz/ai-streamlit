#!/usr/bin/env python
"""Test script for cluster labeling analytics."""

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

# Process first category only for testing
try:
    category = next(iter(merged_clusters_dict.keys()))
    print(f"Processing test category: {category}")
    
    clustering_output_temp = merged_clusters_dict[category]
    
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
        print("No ID columns available in clustering output. Cannot continue.")
        exit(1)
    
    clustering_output = clustering_output_temp[available_id_cols].drop_duplicates().copy()
    
    # Apply special fixes for slash-based categories if needed
    fixed_category = fix_slashes.get(category, category)
    df_grouped_cat = df_grouped[df_grouped["CAT_DSC"] == fixed_category]
    
    if df_grouped_cat.empty:
        print(f"Warning: No data for category {fixed_category} in df_grouped")
        print(f"Available categories: {df_grouped['CAT_DSC'].unique().tolist()}")
    
    # Merge with clustering_output on STORE_NBR
    df_final_cat = clustering_output.merge(df_grouped_cat, on="STORE_NBR", how="left")
    print(f"Merged output has {df_final_cat.shape[0]} rows")
    
    # Test successful
    print("Test completed successfully!")
except Exception as e:
    print(f"Error processing category: {str(e)}")
    raise 