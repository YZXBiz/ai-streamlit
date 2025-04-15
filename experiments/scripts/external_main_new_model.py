# =============================================================================
# 1) Standard Library Imports
# =============================================================================
import sys
import os
import logging
from datetime import datetime

# =============================================================================
# 2) Third-Party Imports
# =============================================================================
import pandas as pd
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------
# Ensure the repo root is on sys.path so "configs" and "utils" modules can be found
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# =============================================================================
# 3) Import Config Variables
# =============================================================================
from configs.config import (
    cat_onehot,
    cat_ordinal,
    min_n_clusters_to_try,
    max_n_clusters_to_try,
    small_cluster_threshold,
    pca_n_features_threshold,
    max_n_pca_features,
    cumulative_variance_target,
    random_state,
    max_iter,
    FEATURE_RANKING_RESULTS_DIR
)

# =============================================================================
# 4) Import Pipeline Utilities
# =============================================================================
from utils.pipeline_utils import (
    preprocess_data,
    drop_correlated_features,
    apply_pca,
    run_kmeans_and_evaluate
)
from utils.external_data_all_prep import load_and_clean_data
from utils.externa_feature_ranking import (
    get_top_80pct_features,
    get_latest_grouped_shap_file
)

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Category-Specific Keep Features
# ------------------------------------------------------------------------------
CATEGORY_SPECIFIC_KEEP_FEATURES = {
    "ORAL HYGIENE": ["WHITE_PCT_PZ_PG",
                     "BLACK_PCT_PZ_PG",
                     "ASIAN_PCT_PZ_PG",
                     "HISP_PCT_PZ_PG",
                     "OTHER_RACE_PCT_PZ_PG"]
}

# =============================================================================
# main() - Orchestrates the entire external clustering process
# =============================================================================
def main():
    logging.info("Starting the main clustering script...")

    # -------------------------------------------------------------------------
    # (A) Load & clean the base data
    # -------------------------------------------------------------------------
    df = load_and_clean_data()
    logging.info(f"Data loaded. Shape: {df.shape}")

    # -------------------------------------------------------------------------
    # (B) Read the latest grouped_shap_all_ file
    # -------------------------------------------------------------------------
    shap_file_path = get_latest_grouped_shap_file(FEATURE_RANKING_RESULTS_DIR)
    logging.info(f"Using SHAP file: {shap_file_path}")

    df_raw = pd.read_excel(shap_file_path)

    # -------------------------------------------------------------------------
    # (C) Get top 80% SHAP features **within each CATEGORY**
    # -------------------------------------------------------------------------
    external_granularity_features_df = (
        df_raw
        .groupby("CATEGORY", group_keys=False)
        .apply(get_top_80pct_features)
        .reset_index(drop=True)
    )

    # Sanity check: each CATEGORY slice should sum to ~0.8
    check_sums = (
        external_granularity_features_df
        .groupby("CATEGORY")["normalized_mean_abs_shap"]
        .sum()
        .reset_index(name="sum_shap")
    )
    tolerance = 1e-8
    if not all(abs(check_sums["sum_shap"] - 0.8) < tolerance):
        raise ValueError("Some categories do not sum to 0.8. Please check your SHAP data.")

    # -------------------------------------------------------------------------
    # (D) Create a dictionary of DataFrames by CATEGORY (top-80%-SHAP features)
    # -------------------------------------------------------------------------
    unique_categories = external_granularity_features_df["CATEGORY"].unique()
    grouped_dfs = {}

    for cat in unique_categories:
        # Fetch the top 80% SHAP features for this category
        features_for_cat = external_granularity_features_df.loc[
            external_granularity_features_df["CATEGORY"] == cat,
            "feature"
        ].unique()

        # Check if we have category-specific keep features
        cat_keep_features = CATEGORY_SPECIFIC_KEEP_FEATURES.get(cat, [])

        # Combine top-80% SHAP features with category-specific keep features
        final_feature_list = list(set(features_for_cat).union(cat_keep_features))

        # Filter the main df to keep only these features + 'STORE_NBR'
        cols_to_keep = [c for c in df.columns if c in final_feature_list]
        
        # Ensure 'STORE_NBR' is always included
        if 'STORE_NBR' not in cols_to_keep:
            cols_to_keep.append('STORE_NBR')

        # Create the subset DataFrame for this category
        grouped_dfs[cat] = df[cols_to_keep].copy()

    # -------------------------------------------------------------------------
    # (E) Prepare a timestamped output folder
    # -------------------------------------------------------------------------
    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"External_Clustering_Run_{run_ts}"
    temp_path = "/home/jovyan/fsassortment/store_clustering/content_external"
    output_path = os.path.join(temp_path, folder_name)
    os.makedirs(output_path, exist_ok=True)

    # -------------------------------------------------------------------------
    # (F) Clustering loop for each category
    # -------------------------------------------------------------------------
    for category_name, df_subset in grouped_dfs.items():
        logging.info(f"Running clustering for category: {category_name}")

        # Identify the subset columns
        subset_cols = df_subset.columns
        
        # (F.1) Preprocess data
        data, pipeline = preprocess_data(
            df_subset,
            cat_onehot,
            cat_ordinal,
            subset_cols
        )

        # (F.2) Drop correlated features
        # Pass in the category-specific keep features so they won't get dropped
        cat_keep_features = CATEGORY_SPECIFIC_KEEP_FEATURES.get(category_name, [])
        data_reduced = drop_correlated_features(
            data,
            corr_threshold=0.8,
            features_to_keep=cat_keep_features
        )

        # (F.3) Apply PCA if needed
        data_pca, pca_model = apply_pca(
            data_reduced,
            pca_n_features_threshold=pca_n_features_threshold,
            max_n_pca_features=max_n_pca_features,
            cumulative_variance_target=cumulative_variance_target,
            random_state=random_state
        )

        # (F.4) Run K-Means to find best_k
        results_df, best_k_val = run_kmeans_and_evaluate(
            data_pca,
            min_n_clusters_to_try=min_n_clusters_to_try,
            max_n_clusters_to_try=max_n_clusters_to_try,
            small_cluster_threshold=small_cluster_threshold,
            random_state=random_state,
            max_iter=max_iter
        )

        # Implement the retry logic
        if best_k_val is None:
            logging.warning(
                f"No valid clusters found for '{category_name}' with "
                f"small_cluster_threshold={small_cluster_threshold}. "
                "Retrying with small_cluster_threshold=20..."
            )
            retry_threshold = 20
            results_df, best_k_val = run_kmeans_and_evaluate(
                data_pca,
                min_n_clusters_to_try=min_n_clusters_to_try,
                max_n_clusters_to_try=max_n_clusters_to_try,
                small_cluster_threshold=retry_threshold,
                random_state=random_state,
                max_iter=max_iter
            )
            if best_k_val is None:
                logging.warning(
                    f"No valid clusters found for '{category_name}' even after "
                    f"lowering threshold to {retry_threshold}. Skipping cluster labeling."
                )
                safe_cat_name = str(category_name).replace(" ", "_").replace("/", "_")
                results_filename = f"clustering_info_{safe_cat_name}_{run_ts}.csv"
                results_path = os.path.join(output_path, results_filename)
                results_df.to_csv(results_path, index=False)
                logging.info(f"Saved results DataFrame for '{category_name}' to {results_path}")
                continue

        # (F.5) Save the clustering info (results_df)
        safe_cat_name = str(category_name).replace(" ", "_").replace("/", "_")
        results_filename = f"clustering_info_{safe_cat_name}_{run_ts}.csv"
        results_path = os.path.join(output_path, results_filename)
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved results DataFrame for '{category_name}' to {results_path}")

        logging.info(f"Best K for {category_name} is: {best_k_val}")

        # (F.6) If still no valid best_k, skip labeling
        if best_k_val is None:
            logging.warning(
                f"No valid clusters found for '{category_name}'. Skipping cluster labeling."
            )
            continue

        # (F.7) Fit K-Means ONCE on the same data used to pick best_k
        kmeans_model = KMeans(
            n_clusters=best_k_val,
            random_state=random_state,
            max_iter=max_iter,
            n_init="auto"
        )
        labels = kmeans_model.fit_predict(data_pca)

        # Attach labels to the original subset
        df_clustered = df_subset.copy()
        df_clustered["cluster_label"] = labels

        # (F.8) Save the cluster-labeled DataFrame
        clustered_filename = f"df_clustered_{safe_cat_name}_{run_ts}.csv"
        clustered_path = os.path.join(output_path, clustered_filename)
        print("-----\n")
        print(clustered_path)
        print("-----\n")
        df_clustered.to_csv(clustered_path, index=False)
        logging.info(f"Saved clustered DataFrame for '{category_name}' to {clustered_path}")

    logging.info("All categories have been processed successfully.")


# ------------------------------------------------------------------------------
# 5) ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()





# # =============================================================================
# # 1) Standard Library Imports
# # =============================================================================
# import sys
# import os
# import logging
# from datetime import datetime

# # =============================================================================
# # 2) Third-Party Imports
# # =============================================================================
# import pandas as pd
# from sklearn.cluster import KMeans

# # ------------------------------------------------------------------------------
# # Ensure the repo root is on sys.path so "configs" and "utils" modules can be found
# # ------------------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root = os.path.abspath(os.path.join(current_dir, ".."))
# if repo_root not in sys.path:
#     sys.path.append(repo_root)

# # =============================================================================
# # 3) Import Config Variables
# # =============================================================================
# from configs.config import (
#     cat_onehot,
#     cat_ordinal,
#     min_n_clusters_to_try,
#     max_n_clusters_to_try,
#     small_cluster_threshold,
#     pca_n_features_threshold,
#     max_n_pca_features,
#     cumulative_variance_target,
#     random_state,
#     max_iter,
#     FEATURE_RANKING_RESULTS_DIR
# )

# # =============================================================================
# # 4) Import Pipeline Utilities
# # =============================================================================
# from utils.pipeline_utils import (
#     preprocess_data,
#     drop_correlated_features,
#     apply_pca,
#     run_kmeans_and_evaluate
# )
# from utils.external_data_all_prep import load_and_clean_data
# from utils.externa_feature_ranking import (
#     get_top_80pct_features,
#     get_latest_grouped_shap_file
# )

# # ------------------------------------------------------------------------------
# # Logging Configuration
# # ------------------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)

# # =============================================================================
# # main() - Orchestrates the entire external clustering process
# # =============================================================================
# def main():
#     logging.info("Starting the main clustering script...")

#     # -------------------------------------------------------------------------
#     # (A) Load & clean the base data (instead of doing large merges here)
#     # -------------------------------------------------------------------------
#     df = load_and_clean_data()
#     logging.info(f"Data loaded. Shape: {df.shape}")

#     # -------------------------------------------------------------------------
#     # (B) Read the latest grouped_shap_all_ file from the newest Ranking_Run_ folder
#     # -------------------------------------------------------------------------
#     shap_file_path = get_latest_grouped_shap_file(FEATURE_RANKING_RESULTS_DIR)
#     logging.info(f"Using SHAP file: {shap_file_path}")

#     df_raw = pd.read_excel(shap_file_path)

#     # -------------------------------------------------------------------------
#     # (C) Get top 80% SHAP features **within each CATEGORY**
#     # -------------------------------------------------------------------------
#     external_granularity_features_df = (
#         df_raw
#         .groupby("CATEGORY", group_keys=False)
#         .apply(get_top_80pct_features)
#         .reset_index(drop=True)
#     )

#     # Sanity check: each CATEGORY slice should sum to ~0.8
#     check_sums = (
#         external_granularity_features_df
#         .groupby("CATEGORY")["normalized_mean_abs_shap"]
#         .sum()
#         .reset_index(name="sum_shap")
#     )
#     tolerance = 1e-8
#     if not all(abs(check_sums["sum_shap"] - 0.8) < tolerance):
#         raise ValueError("Some categories do not sum to 0.8. Please check your SHAP data.")

#     # -------------------------------------------------------------------------
#     # (D) Create a dictionary of DataFrames by CATEGORY (top-80%-SHAP features)
#     # -------------------------------------------------------------------------
#     unique_categories = external_granularity_features_df["CATEGORY"].unique()
#     grouped_dfs = {}

#     for cat in unique_categories:
#         # Fetch the top 80% SHAP features for this category
#         features_for_cat = external_granularity_features_df.loc[
#             external_granularity_features_df["CATEGORY"] == cat,
#             "feature"
#         ].unique()

#         # Filter the main df to keep only these features + 'STORE_NBR'
#         cols_to_keep = [c for c in df.columns if c in features_for_cat]
#         grouped_dfs[cat] = df[cols_to_keep + ['STORE_NBR']].copy()

#     # -------------------------------------------------------------------------
#     # (E) Prepare a timestamped output folder
#     # -------------------------------------------------------------------------
#     run_ts = datetime.now().strftime("%Y%m%d_%H%M")
#     folder_name = f"External_Clustering_Run_{run_ts}"
#     output_path = os.path.join("content_external", folder_name)
#     os.makedirs(output_path, exist_ok=True)

#     # -------------------------------------------------------------------------
#     # (F) Clustering loop for each category
#     # -------------------------------------------------------------------------
#     for category_name, df_subset in grouped_dfs.items():
#         logging.info(f"Running clustering for category: {category_name}")

#         # Identify the subset columns
#         subset_cols = df_subset.columns

#         # (F.1) Preprocess data
#         data, pipeline = preprocess_data(
#             df_subset,
#             cat_onehot,
#             cat_ordinal,
#             subset_cols
#         )

#         # (F.2) Drop correlated features
#         data_reduced = drop_correlated_features(data, corr_threshold=0.8)

#         # (F.3) Apply PCA if needed
#         data_pca, pca_model = apply_pca(
#             data_reduced,
#             pca_n_features_threshold=pca_n_features_threshold,
#             max_n_pca_features=max_n_pca_features,
#             cumulative_variance_target=cumulative_variance_target,
#             random_state=random_state
#         )

#         # (F.4) Run K-Means to find best_k, and collect results
#         results_df, best_k_val = run_kmeans_and_evaluate(
#             data_pca,
#             min_n_clusters_to_try=min_n_clusters_to_try,
#             max_n_clusters_to_try=max_n_clusters_to_try,
#             small_cluster_threshold=small_cluster_threshold,
#             random_state=random_state,
#             max_iter=max_iter
#         )

#         # ---------------------------------------------------------------------
#         # Implement the retry logic with small_cluster_threshold=20
#         # ---------------------------------------------------------------------
#         if best_k_val is None:
#             logging.warning(
#                 f"No valid clusters found for '{category_name}' with "
#                 f"small_cluster_threshold={small_cluster_threshold}. "
#                 "Retrying with small_cluster_threshold=20..."
#             )
#             retry_threshold = 20
#             results_df, best_k_val = run_kmeans_and_evaluate(
#                 data_pca,
#                 min_n_clusters_to_try=min_n_clusters_to_try,
#                 max_n_clusters_to_try=max_n_clusters_to_try,
#                 small_cluster_threshold=retry_threshold,
#                 random_state=random_state,
#                 max_iter=max_iter
#             )

#             # If still no valid clusters, skip final labeling
#             if best_k_val is None:
#                 logging.warning(
#                     f"No valid clusters found for '{category_name}' even after "
#                     f"lowering threshold to {retry_threshold}. Skipping cluster labeling."
#                 )
#                 # Save the results DataFrame anyway
#                 safe_cat_name = str(category_name).replace(" ", "_").replace("/", "_")
#                 results_filename = f"clustering_info_{safe_cat_name}_{run_ts}.csv"
#                 results_path = os.path.join(output_path, results_filename)
#                 results_df.to_csv(results_path, index=False)
#                 logging.info(f"Saved results DataFrame for '{category_name}' to {results_path}")
#                 continue

#         # ---------------------------------------------------------------------
#         # (F.5) Save the clustering info (results_df) for the final attempt
#         # ---------------------------------------------------------------------
#         safe_cat_name = str(category_name).replace(" ", "_").replace("/", "_")
#         results_filename = f"clustering_info_{safe_cat_name}_{run_ts}.csv"
#         results_path = os.path.join(output_path, results_filename)
#         results_df.to_csv(results_path, index=False)
#         logging.info(f"Saved results DataFrame for '{category_name}' to {results_path}")

#         logging.info(f"Best K for {category_name} is: {best_k_val}")

#         # (F.6) If no valid best_k, skip cluster labeling
#         if best_k_val is None:
#             logging.warning(
#                 f"No valid clusters found for '{category_name}'. Skipping cluster labeling."
#             )
#             continue

#         # ---------------------------------------------------------------------
#         # (F.7) Fit K-Means ONCE on the same data used to pick best_k
#         # ---------------------------------------------------------------------
#         kmeans_model = KMeans(
#             n_clusters=best_k_val,
#             random_state=random_state,
#             max_iter=max_iter,
#             n_init="auto"
#         )
#         labels = kmeans_model.fit_predict(data_pca)

#         # Attach labels to the original subset
#         df_clustered = df_subset.copy()
#         df_clustered["cluster_label"] = labels

#         # (F.8) Save the cluster-labeled DataFrame
#         clustered_filename = f"df_clustered_{safe_cat_name}_{run_ts}.csv"
#         clustered_path = os.path.join(output_path, clustered_filename)
#         df_clustered.to_csv(clustered_path, index=False)
#         logging.info(f"Saved clustered DataFrame for '{category_name}' to {clustered_path}")

#     logging.info("All categories have been processed successfully.")


# # ------------------------------------------------------------------------------
# # 5) ENTRY POINT
# # ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()


