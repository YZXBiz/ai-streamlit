# =============================================================================
# 1) Standard Library Imports
# =============================================================================
import sys
import os
import logging
import re
from datetime import datetime

# =============================================================================
# 2) Third-Party Imports
# =============================================================================
import pandas as pd

# -----------------------------------------------------------------------------
# 3) Ensure the repo root is on sys.path so we can import modules such as configs, utils
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# =============================================================================
# 4) Project-Specific Imports
# =============================================================================
from configs.internal_config import INTERNAL_CDTS

# Merging functions from merging_centroid_utils
from utils.merging_centroid_utils import (
    load_datasets,
    extract_granularities,
    merge_cluster_labels,
    build_feature_matrix,
    merge_and_scale,
    reassign_small_clusters,
    build_final_df,
    save_final_df,  # Updated with file_suffix
)

from utils.merging_auto_utils import (
    create_merged_output_folder,
    get_clustered_file,
    get_latest_run_folder,
    sanitize_name,
    create_merged_output_folder_all,
)

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)


# =============================================================================
# main() - Orchestrates the merging of internal + external cluster-labeled CSVs
# =============================================================================
def main():
    """
    Main function to:
      - (Optionally) run external_main.py and internal_main.py
      - Find the latest external and internal cluster-labeled CSV folders
      - For each category in INTERNAL_CDTS, find matching CSVs, merge them
        with centroid-based rebalancing, and save the merged output.

      Additionally:
      - Create a second set of merged outputs (the "All" set) in
        Merged_Clustering_Output_Run_All_{YYYYMMDD_HHMM} under merged_content_all,
        augmented with extra columns from df_all_external.
    """

    # --------------------------------------------------------------------------
    # (A) (Optional) Run external_main.py / internal_main.py automatically
    # --------------------------------------------------------------------------
    # logging.info("Step 1: Running external_main.py...")
    # external_main.main()
    #
    # logging.info("Step 2: Running internal_main.py...")
    # internal_main.main()

    # --------------------------------------------------------------------------
    # (B) Locate the newly created folders for external and internal runs
    # --------------------------------------------------------------------------
    content_dir = os.path.join(repo_root, "content_external")
    internal_content_dir = os.path.join(repo_root, "internal_content")

    latest_external_dirname = get_latest_run_folder(content_dir, "External_Clustering_Run_")
    if not latest_external_dirname:
        logging.error("No External_Clustering_Run_ folder found in 'content_external/'. Exiting.")
        return
    latest_external_folder = os.path.join(content_dir, latest_external_dirname)

    latest_internal_dirname = get_latest_run_folder(
        internal_content_dir, "Internal_Clustering_Run_"
    )
    if not latest_internal_dirname:
        logging.error("No Internal_Clustering_Run_ folder found in 'internal_content/'. Exiting.")
        return
    latest_internal_folder = os.path.join(internal_content_dir, latest_internal_dirname)

    logging.info(f"Found latest external folder: {latest_external_folder}")
    logging.info(f"Found latest internal folder: {latest_internal_folder}")

    # --------------------------------------------------------------------------
    # (C) Prepare a time-stamped folder for all merged outputs
    # --------------------------------------------------------------------------
    merged_content_dir = os.path.join(repo_root, "merged_content")
    merged_output_dir, merged_run_ts = create_merged_output_folder(merged_content_dir)
    logging.info(f"Will save standard merged outputs to: {merged_output_dir}")

    # --------------------------------------------------------------------------
    # (C.1) Prepare a second time-stamped folder for the 'All' merged outputs
    # --------------------------------------------------------------------------
    merged_content_all_dir = os.path.join(repo_root, "merged_content_all")
    merged_output_dir_all, merged_run_ts_all = create_merged_output_folder_all(
        merged_content_all_dir
    )
    logging.info(f"Will also save 'All' merged outputs to: {merged_output_dir_all}")

    # --------------------------------------------------------------------------
    # (D) Load the big external dataset containing extra columns
    # --------------------------------------------------------------------------
    df_all_external = pd.read_csv(
        "/home/jovyan/fsassortment/store_clustering/data/clustering_features_plus_night_traffic_07022025.csv"
    )

    # --------------------------------------------------------------------------
    # (E) For each category in INTERNAL_CDTS, find matching CSVs, then merge
    # --------------------------------------------------------------------------
    for cat in INTERNAL_CDTS:
        cat_sanitized = sanitize_name(cat)
        logging.info(f"Processing category '{cat}' (sanitized: '{cat_sanitized}') ...")

        # (E.1) Locate the cluster-labeled CSVs in the internal and external folders
        internal_csv = get_clustered_file(latest_internal_folder, cat_sanitized)
        external_csv = get_clustered_file(latest_external_folder, cat_sanitized)

        # Skip if missing either internal or external file
        if not internal_csv:
            logging.warning(
                f"[SKIP] No internal CSV found for '{cat}' (sanitized '{cat_sanitized}')"
            )
            continue
        if not external_csv:
            logging.warning(
                f"[SKIP] No external CSV found for '{cat}' (sanitized '{cat_sanitized}')"
            )
            continue

        logging.info(f"Found internal CSV: {internal_csv}")
        logging.info(f"Found external CSV: {external_csv}")

        # ----------------------------------------------------------------------
        # (F) Merge them with centroid rebalancing logic
        # ----------------------------------------------------------------------
        # (F.1) Load Datasets
        int_df, ext_df = load_datasets(internal_csv, external_csv)

        # (F.2) Extract "granularity" from filenames
        internal_granularity, external_granularity = extract_granularities(
            internal_csv, external_csv
        )

        # (F.3) Merge cluster labels
        df_merged_clusters = merge_cluster_labels(
            int_df, ext_df, internal_granularity, external_granularity
        )

        # (F.4) Build feature matrix (dropping cluster labels)
        df_features = build_feature_matrix(int_df, ext_df)

        # (F.5) Merge cluster labels with features, scale numeric columns
        df_scaled = merge_and_scale(df_merged_clusters, df_features)

        # (F.6) Reassign "small" clusters (<100) to nearest large centroid
        df_rebalanced = reassign_small_clusters(df_scaled, min_cluster_size=100)

        # (F.7) Build final merged DataFrame with rebalanced labels
        final_df = build_final_df(df_rebalanced, int_df, ext_df)

        # (F.8) Save final DataFrame (standard version, no suffix)
        output_path = save_final_df(
            final_df,
            internal_granularity,
            external_granularity,
            merged_output_dir,
            file_suffix="",  # No suffix for standard version
        )
        logging.info(f"Merged & rebalanced file saved: {output_path}")

        # ----------------------------------------------------------------------
        # (G) Create the "All" version by adding new columns from df_all_external
        #     via an inner merge on str_nbr (final_df) and STORE_NBR (df_all_external)
        # ----------------------------------------------------------------------
        # We first isolate columns in df_all_external that are not already in final_df
        columns_to_add = df_all_external.columns.difference(final_df.columns).tolist()
        # Ensure we include STORE_NBR for the join
        if "STORE_NBR" not in columns_to_add:
            columns_to_add.insert(0, "STORE_NBR")

        df_all_external_subset = df_all_external[columns_to_add]

        # Now merge on str_nbr (from final_df) == STORE_NBR (from df_all_external)
        final_df_all = pd.merge(
            final_df, df_all_external_subset, how="inner", left_on="store_nbr", right_on="STORE_NBR"
        )
        ### ------------------------------------------------------------------------------------------
        # Fetch data from the SQL query
        path = "/home/jovyan/fsassortment/store_clustering/data/df_region.parquet"
        df_region = pd.read_parquet(path)

        # Define the mapping for time zones to regions
        time_zone_to_region = {
            "Mountain AZ": "west",
            "Mountain": "west",
            "Hawaii-Aleutian": "west",
            "Pacific": "west",
            "Alaska": "west",
            "Eastern": "nonwest",
            "Central": "nonwest",
            "Atlantic-PR": "nonwest",
        }

        # Map the time zones to regions in df_need_states
        df_region["region"] = df_region["TIME_ZONE_ID"].map(time_zone_to_region)

        # Handle cases where the time zone is not in the mapping
        df_region["region"] = df_region["region"].fillna("unknown")
        df_region.columns = df_region.columns.str.lower()

        # Merge the dataframes
        final_df_all = pd.merge(final_df_all, df_region, on="store_nbr")

        # Perform the concatenation only if the left-hand side of the split is 'GROCERY'
        final_df_all["rebalanced_demand_cluster_labels"] = final_df_all.apply(
            lambda row: row["rebalanced_demand_cluster_labels"] + "_" + row["region"]
            if row["external_granularity"].split("_", 1)[0] == "GROCERY"
            else row["rebalanced_demand_cluster_labels"],
            axis=1,
        )

        # Drop unnecessary columns
        final_df_all = final_df_all.drop(["region", "time_zone_id"], axis=1)

        # Print or return the final DataFrame
        print(final_df_all)
        ### ------------------------------------------------------------------------------------------
        # Save final_df_all (with "ALL" suffix)
        output_path_all = save_final_df(
            final_df_all,
            internal_granularity,
            external_granularity,
            merged_output_dir_all,
            file_suffix="ALL",
        )
        logging.info(f"All-included merged file saved: {output_path_all}")

    logging.info("All merges completed successfully.")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main()


# # =============================================================================
# # 1) Standard Library Imports
# # =============================================================================
# import sys
# import os
# import logging
# import re
# from datetime import datetime

# # =============================================================================
# # 2) Third-Party Imports
# # =============================================================================
# import pandas as pd

# # -----------------------------------------------------------------------------
# # 3) Ensure the repo root is on sys.path so we can import modules such as configs, utils
# # -----------------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root = os.path.abspath(os.path.join(current_dir, ".."))
# if repo_root not in sys.path:
#     sys.path.append(repo_root)

# # =============================================================================
# # 4) Project-Specific Imports
# # =============================================================================
# from configs.internal_config import INTERNAL_CDTS

# # Merging functions from merging_centroid_utils
# from utils.merging_centroid_utils import (
#     load_datasets,
#     extract_granularities,
#     merge_cluster_labels,
#     build_feature_matrix,
#     merge_and_scale,
#     reassign_small_clusters,
#     build_final_df,
#     save_final_df
# )

# from utils.merging_auto_utils import (
#     create_merged_output_folder,
#     get_clustered_file,
#     get_latest_run_folder,
#     sanitize_name
# )

# # -----------------------------------------------------------------------------
# # Logging Setup
# # -----------------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)

# # =============================================================================
# # main() - Orchestrates the merging of internal + external cluster-labeled CSVs
# # =============================================================================
# def main():
#     """
#     Main function to:
#       - (Optionally) run external_main.py and internal_main.py
#       - Find the latest external and internal cluster-labeled CSV folders
#       - For each category in INTERNAL_CDTS, find matching CSVs, merge them
#         with centroid-based rebalancing, and save the merged output.
#     """

#     # --------------------------------------------------------------------------
#     # (A) (Optional) Run external_main.py / internal_main.py automatically
#     # --------------------------------------------------------------------------
#     # logging.info("Step 1: Running external_main.py...")
#     # external_main.main()
#     #
#     # logging.info("Step 2: Running internal_main.py...")
#     # internal_main.main()

#     # --------------------------------------------------------------------------
#     # (B) Locate the newly created folders for external and internal runs
#     # --------------------------------------------------------------------------
#     content_dir = os.path.join(repo_root, "content_external")
#     internal_content_dir = os.path.join(repo_root, "internal_content")
#     merged_content_dir = os.path.join(repo_root, "merged_content")

#     latest_external_dirname = get_latest_run_folder(content_dir, "External_Clustering_Run_")
#     if not latest_external_dirname:
#         logging.error("No External_Clustering_Run_ folder found in 'content_external/'. Exiting.")
#         return
#     latest_external_folder = os.path.join(content_dir, latest_external_dirname)

#     latest_internal_dirname = get_latest_run_folder(internal_content_dir, "Internal_Clustering_Run_")
#     if not latest_internal_dirname:
#         logging.error("No Internal_Clustering_Run_ folder found in 'internal_content/'. Exiting.")
#         return
#     latest_internal_folder = os.path.join(internal_content_dir, latest_internal_dirname)

#     logging.info(f"Found latest external folder: {latest_external_folder}")
#     logging.info(f"Found latest internal folder: {latest_internal_folder}")

#     # --------------------------------------------------------------------------
#     # (C) Prepare a time-stamped folder for all merged outputs
#     # --------------------------------------------------------------------------
#     merged_output_dir, merged_run_ts = create_merged_output_folder(merged_content_dir)
#     logging.info(f"Will save merged outputs to: {merged_output_dir}")

#     # --------------------------------------------------------------------------
#     # (D) For each category in INTERNAL_CDTS, find matching CSVs, then merge
#     # --------------------------------------------------------------------------
#     for cat in INTERNAL_CDTS:
#         cat_sanitized = sanitize_name(cat)
#         logging.info(f"Processing category '{cat}' (sanitized: '{cat_sanitized}') ...")

#         # (D.1) Locate the cluster-labeled CSVs in the internal and external folders
#         internal_csv = get_clustered_file(latest_internal_folder, cat_sanitized)
#         external_csv = get_clustered_file(latest_external_folder, cat_sanitized)

#         # Skip if missing either internal or external file
#         if not internal_csv:
#             logging.warning(f"[SKIP] No internal CSV found for '{cat}' (sanitized '{cat_sanitized}')")
#             continue
#         if not external_csv:
#             logging.warning(f"[SKIP] No external CSV found for '{cat}' (sanitized '{cat_sanitized}')")
#             continue

#         logging.info(f"Found internal CSV: {internal_csv}")
#         logging.info(f"Found external CSV: {external_csv}")

#         # ----------------------------------------------------------------------
#         # (E) Merge them with centroid rebalancing logic
#         # ----------------------------------------------------------------------
#         # (E.1) Load Datasets
#         int_df, ext_df = load_datasets(internal_csv, external_csv)

#         # (E.2) Extract "granularity" from filenames
#         internal_granularity, external_granularity = extract_granularities(internal_csv, external_csv)

#         # (E.3) Merge cluster labels
#         df_merged_clusters = merge_cluster_labels(int_df, ext_df, internal_granularity, external_granularity)

#         # (E.4) Build feature matrix (dropping cluster labels)
#         df_features = build_feature_matrix(int_df, ext_df)

#         # (E.5) Merge cluster labels with features, scale numeric columns
#         df_scaled = merge_and_scale(df_merged_clusters, df_features)

#         # (E.6) Reassign "small" clusters (<100) to nearest large centroid
#         df_rebalanced = reassign_small_clusters(df_scaled, min_cluster_size=100)

#         # (E.7) Build final merged DataFrame with rebalanced labels
#         final_df = build_final_df(df_rebalanced, int_df, ext_df)

#         # (E.8) Save final DataFrame to the new folder
#         output_path = save_final_df(
#             final_df,
#             internal_granularity,
#             external_granularity,
#             merged_output_dir
#         )
#         logging.info(f"Merged & rebalanced file saved: {output_path}")

#     logging.info("All merges completed successfully.")


# # =============================================================================
# # Script Entry Point
# # =============================================================================
# if __name__ == "__main__":
#     main()
