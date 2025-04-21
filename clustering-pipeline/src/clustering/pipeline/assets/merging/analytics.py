"""Analytics assets for the clustering pipeline."""

import dagster as dg
import polars as pl
import pandas as pd
import os
import pickle
from datetime import datetime
from typing import Optional, Dict, List, Any


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="analytics",
    group_name="merging",
    deps=["save_merged_cluster_assignments"],
    required_resource_keys={
        "merged_cluster_assignments", 
        "job_params", 
        "ns_map", 
        "ns_sales"
    },
)
def cluster_labeling_analytics(
    context: dg.AssetExecutionContext,
) -> None:
    """Generate analytics and reports based on cluster assignments and need states data.

    Processes cluster assignments along with need states sales data and mappings,
    creates category-specific analytics, and exports Excel files with various
    breakdowns by cluster labels.

    Args:
        context: Dagster asset execution context
    """
    context.log.info("Starting cluster labeling analytics generation")
    
    # Get configuration from resources
    job_params = context.resources.job_params
    output_path = job_params.get("output_path", "data/internal")
    context.log.info(f"Using output path: {output_path}")
    
    # Create timestamped output directory
    current_datetime_str = datetime.now().strftime("%d%m%Y_%H%M")
    output_directory = os.path.join(
        output_path, f"Classics_Output_Run_{current_datetime_str}"
    )
    os.makedirs(output_directory, exist_ok=True)
    context.log.info(f"Created output directory: {output_directory}")
    
    # Load merged cluster assignments
    try:
        merged_clusters_reader = context.resources.merged_cluster_assignments
        merged_clusters_dict = merged_clusters_reader.read()
        if not merged_clusters_dict or not isinstance(merged_clusters_dict, dict):
            context.log.error(f"Invalid merged clusters data: {type(merged_clusters_dict)}")
            raise ValueError(f"Invalid merged clusters data: {type(merged_clusters_dict)}")
        
        context.log.info(f"Loaded merged cluster assignments with {len(merged_clusters_dict)} categories")
        context.log.info(f"Categories: {list(merged_clusters_dict.keys())}")
    except Exception as e:
        context.log.error(f"Error loading merged cluster assignments: {str(e)}")
        raise ValueError(f"Could not read merged cluster assignments: {str(e)}")
    
    # Load need states sales data using resource
    try:
        ns_sales_reader = context.resources.ns_sales
        ns_sales_df = ns_sales_reader.read()
        
        # Convert to pandas if it's a polars DataFrame
        if isinstance(ns_sales_df, pl.DataFrame):
            df_need_states = ns_sales_df.to_pandas()
            context.log.info("Converted need states sales data from Polars to Pandas")
        else:
            df_need_states = ns_sales_df
            
        context.log.info(f"Loaded need states sales data with {df_need_states.shape[0]} rows")
        context.log.info(f"Need states columns: {df_need_states.columns.tolist()}")
    except Exception as e:
        context.log.error(f"Error loading need states sales data: {str(e)}")
        raise ValueError(f"Could not read need states sales data: {str(e)}")
    
    # Load need states mapping data using resource
    try:
        ns_map_reader = context.resources.ns_map
        ns_map_df = ns_map_reader.read()
        
        # Convert to pandas if it's a polars DataFrame
        if isinstance(ns_map_df, pl.DataFrame):
            df_need_states_mapping = ns_map_df.to_pandas()
            context.log.info("Converted need states mapping data from Polars to Pandas")
        else:
            df_need_states_mapping = ns_map_df
            
        context.log.info(f"Loaded need states mapping data with {df_need_states_mapping.shape[0]} rows")
        context.log.info(f"Need states mapping columns: {df_need_states_mapping.columns.tolist()}")
    except Exception as e:
        context.log.error(f"Error loading need states mapping data: {str(e)}")
        raise ValueError(f"Could not read need states mapping data: {str(e)}")
    
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
            context.log.warning(f"Missing columns in df_need_states_mapping: {missing_columns}")
            # Add placeholder columns for any missing ones
            for col in missing_columns:
                df_need_states_mapping[col] = "UNKNOWN"
                context.log.info(f"Added placeholder column: {col}")
        
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
        context.log.info("Prepared need states mapping columns")
    except Exception as e:
        context.log.error(f"Error preparing need states mapping: {str(e)}")
        raise ValueError(f"Failed to prepare need states mapping: {str(e)}")
    
    # Merge need states data with mapping
    try:
        rows_before = df_need_states.shape[0]
        
        # Check and rename columns if needed
        if "SKU_NBR" in df_need_states.columns and "PRODUCT_ID" not in df_need_states.columns:
            df_need_states = df_need_states.rename(columns={"SKU_NBR": "PRODUCT_ID"})
            context.log.info("Renamed SKU_NBR column to PRODUCT_ID")
        
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
        context.log.info(f"Merged need states data: {rows_before} rows before, {rows_after} rows after merge")
    except Exception as e:
        context.log.error(f"Error merging need states data: {str(e)}")
        raise ValueError(f"Failed to merge need states data: {str(e)}")
    
    # Create grouped dataset
    try:
        group_columns = [
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
        ]
        
        # Make sure all required columns exist
        missing_group_cols = [col for col in group_columns if col not in df_need_states.columns]
        if missing_group_cols:
            context.log.warning(f"Missing columns for grouping: {missing_group_cols}")
            # Add placeholder columns
            for col in missing_group_cols:
                df_need_states[col] = "UNKNOWN"
                context.log.info(f"Added placeholder column for grouping: {col}")
        
        df_grouped = df_need_states.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )["TOTAL_SALES"].sum()
        
        context.log.info(f"Created grouped dataset with {df_grouped.shape[0]} rows")
    except Exception as e:
        context.log.error(f"Error creating grouped dataset: {str(e)}")
        raise ValueError(f"Failed to create grouped dataset: {str(e)}")
    
    # Dictionary to fix categories that have slashes, etc.
    fix_slashes = {
        "ACNE HSC": "ACNE/HSC",
        "DIET NUTRITION": "DIET/NUTRITION",
        # Add more if needed
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
        """Build attribute-specific dataframes with sales aggregations.
        
        Args:
            df_in: The aggregated dataframe (grouped_internal or grouped_rebalanced)
            cluster_col: 'internal_cluster' or 'rebalanced_cluster'
            attribute_col: 'ATTRIBUTE_1', 'ATTRIBUTE_2', etc.
            
        Returns:
            A new dataframe with aggregated sales data by cluster and attribute
        """
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
            context.log.warning(f"Error building attribute groupings: {str(e)}")
            return None
    
    # Process each category in the merged clusters dictionary
    categories_processed = []
    for category, clustering_output_temp in merged_clusters_dict.items():
        try:
            context.log.info(f"Processing category: {category}")
            
            # Convert Polars DataFrame to Pandas if needed
            if isinstance(clustering_output_temp, pl.DataFrame):
                clustering_output_temp = clustering_output_temp.to_pandas()
                context.log.info("Converted Polars DataFrame to Pandas")
            
            # Check if clustering_output_temp has all required columns
            missing_cols = [col for col in id_cols if col not in clustering_output_temp.columns]
            if missing_cols:
                context.log.warning(f"Missing columns in clustering output: {missing_cols}")
                context.log.info(f"Available columns: {clustering_output_temp.columns.tolist()}")
                # Skip this category if too many columns are missing
                if len(missing_cols) > len(id_cols) / 2:
                    context.log.warning(f"Skipping category {category} due to too many missing columns")
                    continue
            
            # Keep only the ID columns that exist
            available_id_cols = [col for col in id_cols if col in clustering_output_temp.columns]
            if not available_id_cols:
                context.log.warning(f"No ID columns available in clustering output for category {category}. Skipping.")
                continue
            
            clustering_output = clustering_output_temp[available_id_cols].drop_duplicates().copy()
            context.log.info(f"Prepared clustering output with columns: {clustering_output.columns.tolist()}")
            
            # Apply special fixes for slash-based categories if needed
            fixed_category = fix_slashes.get(category, category)
            df_grouped_cat = df_grouped[df_grouped["CAT_DSC"] == fixed_category]
            
            if df_grouped_cat.empty:
                available_categories = df_grouped["CAT_DSC"].unique().tolist()
                context.log.warning(f"No data for category {fixed_category} in df_grouped")
                context.log.info(f"Available categories: {available_categories}")
                
                # Try a case-insensitive match if exact match fails
                similar_categories = [c for c in available_categories if c.lower() == fixed_category.lower()]
                if similar_categories:
                    fixed_category = similar_categories[0]
                    context.log.info(f"Using similar category match: {fixed_category}")
                    df_grouped_cat = df_grouped[df_grouped["CAT_DSC"] == fixed_category]
                else:
                    context.log.warning(f"Skipping category {category} as no matching data found")
                    continue
            
            # Merge with clustering_output on STORE_NBR
            df_final_cat = clustering_output.merge(df_grouped_cat, on="STORE_NBR", how="left")
            context.log.info(f"Merged output has {df_final_cat.shape[0]} rows")
            
            # Check if we have the necessary columns for grouping
            required_cluster_cols = ["internal_cluster", "rebalanced_cluster"]
            missing_required = [col for col in required_cluster_cols if col not in df_final_cat.columns]
            if missing_required:
                context.log.warning(f"Missing required columns for grouping: {missing_required}")
                # Try to create placeholder columns
                for col in missing_required:
                    df_final_cat[col] = "0"  # Default cluster label
                context.log.info(f"Added placeholder columns for missing cluster columns")
            
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
            context.log.info(f"Created internal grouping with {grouped_internal.shape[0]} rows")
            
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
            context.log.info(f"Created rebalanced grouping with {grouped_rebalanced.shape[0]} rows")
            
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
            context.log.info(f"Created {len(internal_attr_sheets)} internal attribute sheets")
            
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
            context.log.info(f"Created {len(rebalanced_attr_sheets)} rebalanced attribute sheets")
            
            # Write to Excel in the output directory
            safe_category_for_filename = fixed_category.replace("/", "_")
            excel_filename = f"classics_output_{safe_category_for_filename}_{current_datetime_str}.xlsx"
            excel_full_path = os.path.join(output_directory, excel_filename)
            
            context.log.info(f"Writing Excel file: {excel_filename}")
            
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
            
            context.log.info(f"Excel file created: {excel_full_path}")
            categories_processed.append(category)
            
            # Print sum checks
            context.log.info(f"Sum checks for category {fixed_category}:")
            context.log.info(
                f"Sum of grouped_internal['TOTAL_SALES_SUM'] (rounded): {round(grouped_internal['TOTAL_SALES_SUM'].sum())}"
            )
            
            # Sum of df_final_cat['TOTAL_SALES']
            if "TOTAL_SALES" in df_final_cat.columns:
                context.log.info(
                    f"Sum of df_final_cat['TOTAL_SALES'] (rounded): {round(df_final_cat['TOTAL_SALES'].sum())}"
                )
            
            # Sum of df_grouped_cat['TOTAL_SALES'] for the clustering stores
            clustering_stores = clustering_output["STORE_NBR"].unique().tolist()
            df_grouped_cat_for_clustering = df_grouped_cat[
                df_grouped_cat["STORE_NBR"].isin(clustering_stores)
            ]
            context.log.info(
                f"Sum of df_grouped_cat['TOTAL_SALES'] for clustering stores (rounded): "
                f"{round(df_grouped_cat_for_clustering['TOTAL_SALES'].sum())}"
            )
            
            # Sum of df_need_states for this category & clustering stores
            category_clustering_sum = df_need_states[
                (df_need_states["CAT_DSC"] == fixed_category)
                & (df_need_states["STORE_NBR"].isin(clustering_stores))
            ]["TOTAL_SALES"].sum()
            context.log.info(
                f"Sum of df_need_states for {fixed_category} & clustering stores (rounded): "
                f"{round(category_clustering_sum)}"
            )
        except Exception as e:
            context.log.error(f"Error processing category {category}: {str(e)}")
            # Continue processing other categories instead of failing entirely
            continue
    
    context.log.info(f"Categories processed: {categories_processed}")
    context.log.info("All categories processed. Cluster labeling analytics complete.")
    
    # Record metadata about the created reports
    return_metadata = {
        "output_directory": output_directory,
        "timestamp": current_datetime_str,
        "categories_processed": categories_processed,
    }
    
    # Add metadata to the asset
    context.add_output_metadata(metadata=return_metadata) 