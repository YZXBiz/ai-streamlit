# internal_main.py
# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import sys
import os
import logging
from datetime import datetime

# ------------------------------------------------------------------------------
# Adjust sys.path so that we can import modules from the project root (../),
# such as configs/ and utils/.
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import pandas as pd
from sklearn.cluster import KMeans

# ------------------------------------------------------------------------------
# Project-specific / local imports
# ------------------------------------------------------------------------------
from configs.internal_config import (
    DATA_DIR,
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
    corr_threshold,
    INTERNAL_CDTS,
    INTERNAL_CLUSTERING_OUTPUT_DIR
)
from utils.pipeline_utils import (
    preprocess_data,
    drop_correlated_features,
    apply_pca,
    run_kmeans_and_evaluate
)
from utils.internal_preprocessing import (
    load_internal_data,
    create_cat_dict,
    sanitize_name
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function that orchestrates the internal data pipeline.
    """
    # ------------------------------------------------------------------------------
    # 1) Load the input data
    # ------------------------------------------------------------------------------
    logging.info("Loading internal data...")
    df_pilot_cat, df_plano, df_NS_inscope = load_internal_data()
    logging.info("Data load complete. Creating cat_dict...")

    # ------------------------------------------------------------------------------
    # 2) Create a dictionary: {category -> pivoted DataFrame with % of SALES by NEED_STATE}
    # ------------------------------------------------------------------------------
    cat_dict = create_cat_dict(df_NS_inscope)
    logging.info("cat_dict created for all in-scope categories from pilot data.")

    # ------------------------------------------------------------------------------
    # Generate a timestamp for output folder creation
    # ------------------------------------------------------------------------------
    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"Internal_Clustering_Run_{run_ts}"
    output_dir = os.path.join(INTERNAL_CLUSTERING_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 3) Loop over each in-scope category and execute the clustering pipeline
    # ------------------------------------------------------------------------------
    for cat in INTERNAL_CDTS:
        # Check if category is found in cat_dict
        if cat not in cat_dict:
            logging.warning(f"Category '{cat}' not found in cat_dict. Skipping.")
            continue

        logging.info(f"Processing category: {cat}")

        # 3.1) Get the pivoted dataframe for this category
        df_clust = cat_dict[cat].copy()

        # 3.2) Identify numeric columns (exclude STORE_NBR if present)
        num_cols = df_clust.drop(columns=["STORE_NBR"], errors="ignore").columns.tolist()

        # 3.3) Preprocess data (scaling/encoding, etc.)
        data, pipeline = preprocess_data(df_clust, cat_onehot, cat_ordinal, num_cols)

        # 3.4) (Optional) Outlier detection is removed; proceed with copy
        data_no_outliers = data.copy()

        # 3.5) Drop highly correlated features
        data_reduced = drop_correlated_features(data_no_outliers, corr_threshold=corr_threshold)

        # 3.6) Apply PCA if needed
        data_pca, pca_model = apply_pca(
            data_reduced,
            pca_n_features_threshold=pca_n_features_threshold,
            max_n_pca_features=max_n_pca_features,
            cumulative_variance_target=cumulative_variance_target,
            random_state=random_state
        )

        # 3.7) Run K-Means for a range of cluster sizes to find the best number of clusters
        results_df, best_k = run_kmeans_and_evaluate(
            data_pca,
            min_n_clusters_to_try,
            max_n_clusters_to_try,
            small_cluster_threshold,
            random_state=random_state,
            max_iter=max_iter
        )

        # ------------------------------------------------------------------------------
        # If no valid cluster solution is found, retry with small_cluster_threshold=10
        # ------------------------------------------------------------------------------
        if best_k is None:
            logging.warning(
                f"No valid clusters found for '{cat}' with "
                f"small_cluster_threshold={small_cluster_threshold}. "
                "Retrying with small_cluster_threshold=10..."
            )
            retry_threshold = 10

            # Re-run clustering with the reduced threshold
            results_df, best_k = run_kmeans_and_evaluate(
                data_pca,
                min_n_clusters_to_try,
                max_n_clusters_to_try,
                retry_threshold,
                random_state=random_state,
                max_iter=max_iter
            )

            # If still no valid clusters, skip
            if best_k is None:
                logging.warning(
                    f"No valid clusters found for '{cat}' even after "
                    f"lowering threshold to {retry_threshold}. Skipping cluster labeling."
                )
                # Save the results DataFrame (optional)
                safe_cat = sanitize_name(cat)
                results_filename = f"clustering_info_{safe_cat}_{run_ts}.csv"
                results_path = os.path.join(output_dir, results_filename)
                results_df.to_csv(results_path, index=False)
                logging.info(f"Saved results DataFrame for '{cat}' to {results_path}")
                continue

        # ------------------------------------------------------------------------------
        # 3.8) We have a valid best_k. Save the clustering metrics (results_df).
        # ------------------------------------------------------------------------------
        safe_cat = sanitize_name(cat)
        results_filename = f"clustering_info_{safe_cat}_{run_ts}.csv"
        results_path = os.path.join(output_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        logging.info(f"Saved results DataFrame for '{cat}' to {results_path}")

        logging.info(f"Optimal K for '{cat}': {best_k}")

        # ------------------------------------------------------------------------------
        # 3.9) Fit K-Means once more on data_pca using the determined best_k
        # ------------------------------------------------------------------------------
        kmeans_model = KMeans(
            n_clusters=best_k,
            random_state=random_state,
            max_iter=max_iter,
            n_init="auto"
        )
        labels = kmeans_model.fit_predict(data_pca)

        # Attach cluster labels to the original pivoted DataFrame
        df_clustered = df_clust.copy()
        df_clustered["cluster_label"] = labels

        # ------------------------------------------------------------------------------
        # 3.10) Save the cluster-labeled DataFrame
        # ------------------------------------------------------------------------------
        clustered_filename = f"df_clustered_{safe_cat}_{run_ts}.csv"
        clustered_path = os.path.join(output_dir, clustered_filename)
        df_clustered.to_csv(clustered_path, index=False)
        logging.info(f"Saved cluster-labeled DataFrame for '{cat}' to {clustered_path}")

    # ------------------------------------------------------------------------------
    # 4) Pipeline complete
    # ------------------------------------------------------------------------------
    logging.info("All INTERNAL_CDTS categories processed successfully.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
# # ------------------------------------------------------------------------------
# # Standard library imports
# # ------------------------------------------------------------------------------
# import sys
# import os
# import logging
# from datetime import datetime

# # ------------------------------------------------------------------------------
# # Adjust sys.path so that we can import modules from the project root (../),
# # such as configs/ and utils/.
# # ------------------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root = os.path.abspath(os.path.join(current_dir, ".."))
# if repo_root not in sys.path:
#     sys.path.append(repo_root)

# # ------------------------------------------------------------------------------
# # Third-party imports
# # ------------------------------------------------------------------------------
# import pandas as pd
# from sklearn.cluster import KMeans

# # ------------------------------------------------------------------------------
# # Project-specific / local imports
# # ------------------------------------------------------------------------------
# from configs.internal_config import (
#     DATA_DIR,
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
#     corr_threshold,
#     INTERNAL_CDTS,
#     INTERNAL_CLUSTERING_OUTPUT_DIR
# )
# from utils.pipeline_utils import (
#     preprocess_data,
#     drop_correlated_features,
#     apply_pca,
#     run_kmeans_and_evaluate
#     # Note: Removed add_cluster_labels import, because we'll label clusters below.
# )
# from utils.internal_preprocessing import (
#     load_internal_data,
#     create_cat_dict,
#     sanitize_name
# )

# # ------------------------------------------------------------------------------
# # Logging configuration
# # ------------------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)

# def main():
#     """
#     Main function that orchestrates the internal data pipeline.
#     """
#     # ------------------------------------------------------------------------------
#     # 1) Load the input data
#     # ------------------------------------------------------------------------------
#     logging.info("Loading internal data...")
#     df_pilot_cat, df_plano, df_NS_inscope = load_internal_data()
#     logging.info("Data load complete. Creating cat_dict...")

#     # ------------------------------------------------------------------------------
#     # 2) Create a dictionary: {category -> pivoted DataFrame with % of SALES by NEED_STATE}
#     # ------------------------------------------------------------------------------
#     cat_dict = create_cat_dict(df_NS_inscope)
#     logging.info("cat_dict created for all in-scope categories from pilot data.")

#     # ------------------------------------------------------------------------------
#     # Generate a timestamp for output folder creation
#     # ------------------------------------------------------------------------------
#     run_ts = datetime.now().strftime("%Y%m%d_%H%M")
#     folder_name = f"Internal_Clustering_Run_{run_ts}"
#     output_dir = os.path.join(INTERNAL_CLUSTERING_OUTPUT_DIR, folder_name)
#     os.makedirs(output_dir, exist_ok=True)

#     # ------------------------------------------------------------------------------
#     # 3) Loop over each in-scope category and execute the clustering pipeline
#     # ------------------------------------------------------------------------------
#     for cat in INTERNAL_CDTS:
#         # Check if category is found in cat_dict
#         if cat not in cat_dict:
#             logging.warning(f"Category '{cat}' not found in cat_dict. Skipping.")
#             continue

#         logging.info(f"Processing category: {cat}")

#         # 3.1) Get the pivoted dataframe for this category
#         df_clust = cat_dict[cat].copy()

#         # 3.2) Identify numeric columns (exclude STORE_NBR if present)
#         num_cols = df_clust.drop(columns=["STORE_NBR"], errors="ignore").columns.tolist()

#         # 3.3) Preprocess data (scaling/encoding, etc.)
#         data, pipeline = preprocess_data(df_clust, cat_onehot, cat_ordinal, num_cols)

#         # 3.4) (Optional) Outlier detection is removed; proceed with copy
#         data_no_outliers = data.copy()

#         # 3.5) Drop highly correlated features
#         data_reduced = drop_correlated_features(data_no_outliers, corr_threshold=corr_threshold)

#         # 3.6) Apply PCA if needed
#         data_pca, pca_model = apply_pca(
#             data_reduced,
#             pca_n_features_threshold=pca_n_features_threshold,
#             max_n_pca_features=max_n_pca_features,
#             cumulative_variance_target=cumulative_variance_target,
#             random_state=random_state
#         )

#         # 3.7) Run K-Means for a range of cluster sizes to find the best number of clusters
#         results_df, best_k = run_kmeans_and_evaluate(
#             data_pca,
#             min_n_clusters_to_try,
#             max_n_clusters_to_try,
#             small_cluster_threshold,
#             random_state=random_state,
#             max_iter=max_iter
#         )

#         # ------------------------------------------------------------------------------
#         # Save the clustering metrics (results_df) with sanitized category name
#         # ------------------------------------------------------------------------------
#         safe_cat = sanitize_name(cat)
#         results_filename = f"clustering_info_{safe_cat}_{run_ts}.csv"
#         results_path = os.path.join(output_dir, results_filename)
#         results_df.to_csv(results_path, index=False)
#         logging.info(f"Saved results DataFrame for '{cat}' to {results_path}")

#         # If best_k is None, skip final clustering
#         if best_k is None:
#             logging.warning(f"No valid clusters found for '{cat}'. Skipping cluster labeling.")
#             continue

#         logging.info(f"Optimal K for '{cat}': {best_k}")

#         # ------------------------------------------------------------------------------
#         # 3.8) Fit K-Means once more on data_pca using the determined best_k
#         # ------------------------------------------------------------------------------
#         kmeans_model = KMeans(
#             n_clusters=best_k,
#             random_state=random_state,
#             max_iter=max_iter,
#             n_init="auto"
#         )
#         labels = kmeans_model.fit_predict(data_pca)

#         # Attach cluster labels to the original pivoted DataFrame
#         df_clustered = df_clust.copy()
#         df_clustered["cluster_label"] = labels

#         # ------------------------------------------------------------------------------
#         # 3.9) Save the cluster-labeled DataFrame
#         # ------------------------------------------------------------------------------
#         clustered_filename = f"df_clustered_{safe_cat}_{run_ts}.csv"
#         clustered_path = os.path.join(output_dir, clustered_filename)
#         df_clustered.to_csv(clustered_path, index=False)
#         logging.info(f"Saved cluster-labeled DataFrame for '{cat}' to {clustered_path}")

#     # ------------------------------------------------------------------------------
#     # 4) Pipeline complete
#     # ------------------------------------------------------------------------------
#     logging.info("All INTERNAL_CDTS categories processed successfully.")


# # ------------------------------------------------------------------------------
# # Entry point
# # ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()
