# internal_main.py
# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import logging
import os
from datetime import datetime

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import pandas as pd

# ------------------------------------------------------------------------------
# Project-specific / local imports
# ------------------------------------------------------------------------------
from configs.internal_config import (
    INTERNAL_CDTS,
    INTERNAL_CLUSTERING_OUTPUT_DIR,
    cat_onehot,
    cat_ordinal,
    corr_threshold,
    cumulative_variance_target,
    max_iter,
    max_n_clusters_to_try,
    max_n_pca_features,
    min_n_clusters_to_try,
    pca_n_features_threshold,
    random_state,
    small_cluster_threshold,
)
from sklearn.cluster import KMeans
from utils.internal_preprocessing import (
    create_cat_dict,
    load_internal_data,
    sanitize_name,
)
from utils.pipeline_utils import (
    apply_pca,
    drop_correlated_features,
    preprocess_data,
    run_kmeans_and_evaluate,
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)


def save_results_df(results_df: pd.DataFrame, cat: str, output_dir: str, run_ts: str) -> str:
    """
    Save clustering results DataFrame to disk.

    Args:
        results_df: DataFrame containing clustering results
        cat: Category name
        output_dir: Directory to save results
        run_ts: Timestamp string

    Returns:
        Path where the results were saved
    """
    safe_cat = sanitize_name(cat)
    results_filename = f"clustering_info_{safe_cat}_{run_ts}.csv"
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved results DataFrame for '{cat}' to {results_path}")
    return results_path


def process_category(
    cat: str,
    cat_dict: dict[str, pd.DataFrame],
    output_dir: str,
    run_ts: str,
    retry_threshold: int = 10,
) -> None:
    """
    Process a single category through the clustering pipeline.

    Args:
        cat: Category name to process
        cat_dict: Dictionary mapping categories to DataFrames
        output_dir: Directory to save results
        run_ts: Timestamp string
        retry_threshold: Threshold to use if initial clustering fails
    """
    # Check if category is found in cat_dict
    if cat not in cat_dict:
        logging.warning(f"Category '{cat}' not found in cat_dict. Skipping.")
        return

    logging.info(f"Processing category: {cat}")

    # Get the pivoted dataframe for this category
    df_clust = cat_dict[cat].copy()

    # Identify numeric columns (exclude STORE_NBR if present)
    num_cols = df_clust.drop(columns=["STORE_NBR"], errors="ignore").columns.tolist()

    # Preprocess data (scaling/encoding, etc.)
    data, pipeline = preprocess_data(df_clust, cat_onehot, cat_ordinal, num_cols)

    # Drop highly correlated features
    data_reduced = drop_correlated_features(data.copy(), corr_threshold=corr_threshold)

    # Apply PCA if needed
    data_pca, _ = apply_pca(
        data_reduced,
        pca_n_features_threshold=pca_n_features_threshold,
        max_n_pca_features=max_n_pca_features,
        cumulative_variance_target=cumulative_variance_target,
        random_state=random_state,
    )

    # Run K-Means for a range of cluster sizes to find the best number of clusters
    results_df, best_k = run_kmeans_and_evaluate(
        data_pca,
        min_n_clusters_to_try,
        max_n_clusters_to_try,
        small_cluster_threshold,
        random_state=random_state,
        max_iter=max_iter,
    )

    # If no valid cluster solution is found, retry with smaller threshold
    if best_k is None:
        logging.warning(
            f"No valid clusters found for '{cat}' with "
            f"small_cluster_threshold={small_cluster_threshold}. "
            f"Retrying with small_cluster_threshold={retry_threshold}..."
        )

        # Re-run clustering with the reduced threshold
        results_df, best_k = run_kmeans_and_evaluate(
            data_pca,
            min_n_clusters_to_try,
            max_n_clusters_to_try,
            retry_threshold,
            random_state=random_state,
            max_iter=max_iter,
        )

        # If still no valid clusters, save results and skip
        if best_k is None:
            logging.warning(
                f"No valid clusters found for '{cat}' even after "
                f"lowering threshold to {retry_threshold}. Skipping cluster labeling."
            )
            save_results_df(results_df, cat, output_dir, run_ts)
            return

    # Save clustering metrics
    save_results_df(results_df, cat, output_dir, run_ts)
    logging.info(f"Optimal K for '{cat}': {best_k}")

    # Fit K-Means once more on data_pca using the determined best_k
    kmeans_model = KMeans(
        n_clusters=best_k, random_state=random_state, max_iter=max_iter, n_init="auto"
    )
    labels = kmeans_model.fit_predict(data_pca)

    # Attach cluster labels to the original pivoted DataFrame
    df_clustered = df_clust.copy()
    df_clustered["cluster_label"] = labels

    # Save the cluster-labeled DataFrame
    safe_cat = sanitize_name(cat)
    clustered_filename = f"df_clustered_{safe_cat}_{run_ts}.csv"
    clustered_path = os.path.join(output_dir, clustered_filename)
    df_clustered.to_csv(clustered_path, index=False)
    logging.info(f"Saved cluster-labeled DataFrame for '{cat}' to {clustered_path}")


def main() -> None:
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
        process_category(cat, cat_dict, output_dir, run_ts)

    # ------------------------------------------------------------------------------
    # 4) Pipeline complete
    # ------------------------------------------------------------------------------
    logging.info("All INTERNAL_CDTS categories processed successfully.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
