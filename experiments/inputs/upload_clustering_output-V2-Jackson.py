#!/usr/bin/env python
"""
Script to upload merged clustering output to Snowflake.
Loads the merged cluster assignments from the pickle file and uploads to Snowflake.
"""

import os
import pandas as pd
import pickle
from snowflake.connector.pandas_tools import write_pandas
from clustering.shared.io.readers.snowflake_reader import SnowflakeReader

# Configuration
SF_DATABASE = "DL_FSCA_SLFSRV"
SF_SCHEMA = "TWA07"
OUTPUT_TABLE_NAME = "FINAL_ASSORTMENT_STORE_CLUSTERS"
ARCHIVE_TABLE_NAME = "FINAL_ASSORTMENT_STORE_CLUSTERS_ARCHIVE"
PICKLE_PATH = "/workspaces/clustering-dagster/data/merging/merged_cluster_assignments.pkl"


def process_clustering_data(pickle_path: str) -> pd.DataFrame:
    """
    Load merged cluster assignments from pickle file,
    process it, and return a standardized DataFrame.
    """
    # Load pickle data
    with open(pickle_path, 'rb') as file:
        data_dict = pickle.load(file)
    
    # Convert dictionary to DataFrame
    dfs = []
    for key, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            dfs.append(data)
        else:
            print(f"Skipping non-DataFrame item with key: {key}")
    
    if not dfs:
        raise ValueError("No valid DataFrames found in pickle file")
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Replace underscores with spaces in category column
    if 'category' in combined_df.columns:
        combined_df['category'] = combined_df['category'].str.replace('_', ' ')
    
    # Create a CAT_DSC column that's identical to category
    combined_df['CAT_DSC'] = combined_df['category']
    
    return combined_df


def upload_to_snowflake(df: pd.DataFrame, timestamp: str = None) -> None:
    """
    Upload the DataFrame to Snowflake tables.
    """
    # Use existing Snowflake configuration from shared implementation
    sf_reader = SnowflakeReader()
    conn = sf_reader._create_connection()
    
    try:
        # Upload to main table
        write_pandas(
            conn,
            df,
            OUTPUT_TABLE_NAME,
            database=SF_DATABASE,
            schema=SF_SCHEMA,
            overwrite=True,
            auto_create_table=True,
        )
        print(f"Data successfully written to {SF_DATABASE}.{SF_SCHEMA}.{OUTPUT_TABLE_NAME}")
        
        # Create archive copy with timestamp
        if timestamp:
            df_archive = df.copy()
            df_archive["timestamp"] = timestamp
            
            write_pandas(
                conn,
                df_archive,
                ARCHIVE_TABLE_NAME,
                database=SF_DATABASE,
                schema=SF_SCHEMA,
                overwrite=False,
                auto_create_table=True,
            )
            print(f"Archive copy with timestamp {timestamp} written to {SF_DATABASE}.{SF_SCHEMA}.{ARCHIVE_TABLE_NAME}")
    finally:
        # Ensure connection is closed
        conn.close()


def main():
    """Main execution function."""
    try:
        # Load and process the data
        processed_df = process_clustering_data(PICKLE_PATH)
        
        # Generate timestamp for archive (current date in YYYYMMDD_HHMM format)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        # Upload to Snowflake
        upload_to_snowflake(processed_df, timestamp)
        
        print(f"Processing complete. Uploaded {processed_df.shape[0]} records across {processed_df['category'].nunique()} categories.")
        
    except Exception as e:
        print(f"Error processing clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
