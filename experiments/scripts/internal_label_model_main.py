# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import sys
import logging
from datetime import datetime
import glob

# =============================================================================
# Third-Party Imports
# =============================================================================
import pandas as pd

# =============================================================================
# 1) Adjust Python path to include the project root
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# =============================================================================
# 2) Import config parameters
# =============================================================================
from configs.config import (
    STORE_COL,
    TARGET_COL,
    COLUMNS_TO_IGNORE,
    TEST_SIZE,
    RANDOM_STATE,
    NUM_LEAVES,
    MAX_DEPTH,
    N_ESTIMATORS,
    LEARNING_RATE,
    STOPPING_ROUNDS,
    RANKING_OUTPUT_DIR,
    DATA_DIR,
    RENAME_DICT
)
from configs.internal_config import INTERNAL_CLUSTERING_OUTPUT_DIR

# =============================================================================
# 3) Import the data loading and ML pipeline utilities
# =============================================================================
from utils.external_data_all_prep import (
    load_and_clean_data,
    load_clustered_csvs
)
from utils.externa_feature_ranking import (
    prepare_data_for_model,
    train_lgbm_multiclass,  # Returns model, top_features_df, acc, class_report, cm_str
    compute_shap_values_per_class,
    get_latest_internal_run_directory
)

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# run_multi_cluster_pipeline
# =============================================================================
def run_multi_cluster_pipeline(
    data_dir: str = DATA_DIR,
    directory: str = None,
    columns_to_ignore: list = COLUMNS_TO_IGNORE,
    core_cols: list = None,
    rename_dict: dict = None,
    output_folder: str = RANKING_OUTPUT_DIR
):
    """
    Multi-cluster pipeline that performs the following steps:
      1) Creates a timestamped subfolder in RANKING_OUTPUT_DIR: "Ranking_Run_{YYYYMMDD}_{HHMM}".
      2) For each 'key' in clustered_data:
         - Merges the cluster_label DataFrame with `new_features_df`.
         - Trains a LightGBM model (captures acc, class_report, confusion_matrix).
         - Computes per-class SHAP => shap_table_long.
         - Groups by feature => mean_abs_shap, adds `normalized_mean_abs_shap`.
         - Writes a grouped file (with all features) to an Excel file.
         - Accumulates metrics in model_metrics_accumulator.
      3) Accumulates shap_table_long for all keys => shap_final_df => writes to Excel.
      4) Concatenates the grouped files for all keys, adds `CATEGORY` (from `key`),
         then writes to a final Excel in the same subfolder.
      5) Saves model_metrics_accumulator to "model_metrics_{YYYYMMDD}_{HHMM}.xlsx" 
         with columns: [key, accuracy, classification_report, confusion_matrix].
    """
    # -------------------------------------------------------------------------
    # Handle default mutable parameters
    # -------------------------------------------------------------------------
    if core_cols is None:
        core_cols = []
    if rename_dict is None:
        rename_dict = {}

    # -------------------------------------------------------------------------
    # Require a valid directory path
    # -------------------------------------------------------------------------
    if directory is None:
        raise ValueError("No directory specified. Please provide a valid path to the clustered data.")

    # -------------------------------------------------------------------------
    # (A) Create a new subfolder for this run
    # -------------------------------------------------------------------------
    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    ranking_run_folder_name = f"Ranking_Run_{now_str}"
    run_folder_path = os.path.join(output_folder, ranking_run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    logger.info(f"Created run folder: {run_folder_path}")

    # -------------------------------------------------------------------------
    # (B) Load data (both clustered data and the main new_features_df)
    # -------------------------------------------------------------------------
    logger.info("Loading clustered_data and new_features_df.")
    clustered_data = load_clustered_csvs(directory)  # Returns dict of DataFrames
    new_features_df = load_and_clean_data()          # Returns DataFrame

    # Prepare accumulators
    shap_accumulator = []           # Will store shap_table_long for each key
    model_metrics_accumulator = []  # Will store model metrics for each key

    # -------------------------------------------------------------------------
    # (C) For each 'key' in clustered_data => train model + create grouped files
    # -------------------------------------------------------------------------
    for key in clustered_data.keys():
        logger.info(f"=== Processing key: {key} ===")

        # Merge cluster_label with new_features
        target_df = clustered_data[key][[STORE_COL, TARGET_COL]].copy()
        model_df = pd.merge(target_df, new_features_df, on=STORE_COL, how="inner")

        # ---------------------------------------------------------------------
        # DEBUG PRINT: Show key + value counts of the target column
        # ---------------------------------------------------------------------
        print(f"\n=== Debug Info for key={key} ===")
        print("Value counts of the target column:")
        print(model_df[TARGET_COL].value_counts())
        print("===================================\n")

        # ---------------------------------------------------------------------
        # Prepare data for model training/testing
        # ---------------------------------------------------------------------
        X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_model(
            model_df=model_df,
            store_col=STORE_COL,
            target_col=TARGET_COL,
            columns_to_ignore=columns_to_ignore,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # ---------------------------------------------------------------------
        # Train model and retrieve metrics
        # ---------------------------------------------------------------------
        # model, top_features_df, accuracy, class_rep, confusion_matrix_str
        model, _, acc, class_rep, cm_str = train_lgbm_multiclass(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_cols=feature_cols,
            num_leaves=NUM_LEAVES,
            max_depth=MAX_DEPTH,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
            stopping_rounds=STOPPING_ROUNDS
        )

        # Add this model's metrics to the accumulator
        model_metrics_accumulator.append({
            "key": key,
            "accuracy": acc,
            "classification_report": class_rep,
            "confusion_matrix": cm_str
        })

        # ---------------------------------------------------------------------
        # Compute per-class SHAP and store results
        # ---------------------------------------------------------------------
        shap_table_long = compute_shap_values_per_class(model, X_test)
        shap_table_long["renamed_feature"] = shap_table_long["feature"].map(rename_dict)\
                                                 .fillna(shap_table_long["feature"])
        shap_table_long["key"] = key
        shap_accumulator.append(shap_table_long)

        # ---------------------------------------------------------------------
        # Group by feature => compute average mean_abs_shap across classes
        # ---------------------------------------------------------------------
        grouped = shap_table_long.groupby("feature", as_index=False)["mean_abs_shap"].mean()
        grouped.sort_values("mean_abs_shap", ascending=False, inplace=True)
        total_shap = grouped["mean_abs_shap"].sum()
        grouped["normalized_mean_abs_shap"] = grouped["mean_abs_shap"] / (total_shap if total_shap != 0 else 1)
        grouped["renamed_feature"] = grouped["feature"].map(rename_dict).fillna(grouped["feature"])
        grouped["key"] = key

        # ---------------------------------------------------------------------
        # Write grouped results to an Excel file
        # ---------------------------------------------------------------------
        grouped_filename = f"grouped_shap_{key}_{now_str}.xlsx"
        grouped_path = os.path.join(run_folder_path, grouped_filename)
        with pd.ExcelWriter(grouped_path) as writer:
            grouped.to_excel(writer, sheet_name="grouped_shap", index=False)
        logger.info(f"Wrote grouped SHAP file for key='{key}' => {grouped_path}")

    # -------------------------------------------------------------------------
    # (D) Combine shap_table_long for all keys => shap_final_df => Excel
    # -------------------------------------------------------------------------
    shap_final_df = pd.concat(shap_accumulator, ignore_index=True)
    shap_filename = f"shap_values_all_{now_str}.xlsx"
    shap_path = os.path.join(run_folder_path, shap_filename)
    with pd.ExcelWriter(shap_path) as writer:
        shap_final_df.to_excel(writer, sheet_name="shap_values", index=False)
    logger.info(f"Saved combined SHAP table for all keys to: {shap_path}")

    # -------------------------------------------------------------------------
    # (E) Concatenate all grouped files for this run, add 'CATEGORY'
    # -------------------------------------------------------------------------
    grouped_pattern = os.path.join(run_folder_path, f"grouped_shap_*_{now_str}.xlsx")
    grouped_files = glob.glob(grouped_pattern)
    grouped_acc = []

    for file_path in grouped_files:
        temp_df = pd.read_excel(file_path)
        # Rename 'key' => 'CATEGORY'
        if "key" in temp_df.columns:
            temp_df.rename(columns={"key": "CATEGORY"}, inplace=True)
        grouped_acc.append(temp_df)

    if grouped_acc:
        combined_grouped = pd.concat(grouped_acc, ignore_index=True)
        combined_filename = f"grouped_shap_all_{now_str}.xlsx"
        combined_path = os.path.join(run_folder_path, combined_filename)
        with pd.ExcelWriter(combined_path) as writer:
            combined_grouped.to_excel(writer, sheet_name="grouped_shap_all", index=False)
        logger.info(f"Wrote concatenated grouped file for all keys => {combined_path}")
    else:
        logger.warning(f"No grouped_shap files found with pattern: {grouped_pattern}")

    # -------------------------------------------------------------------------
    # (F) Write model metrics for each key to a final Excel file
    # -------------------------------------------------------------------------
    if model_metrics_accumulator:
        metrics_df = pd.DataFrame(model_metrics_accumulator)
        metrics_filename = f"model_metrics_{now_str}.xlsx"
        metrics_path = os.path.join(run_folder_path, metrics_filename)
        with pd.ExcelWriter(metrics_path) as writer:
            metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        logger.info(f"Wrote model metrics for all keys => {metrics_path}")
    else:
        logger.warning("No model metrics were collected.")

    logger.info("Run completed successfully.")

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    # Dynamically discover the latest "Internal_Clustering_Run_..." directory
    base_internal_content_dir = INTERNAL_CLUSTERING_OUTPUT_DIR
    latest_internal_path = get_latest_internal_run_directory(base_internal_content_dir)
    
    run_multi_cluster_pipeline(
        data_dir=DATA_DIR,
        directory=latest_internal_path,
        columns_to_ignore=COLUMNS_TO_IGNORE,
        core_cols=[],
        rename_dict=RENAME_DICT,
        output_folder=RANKING_OUTPUT_DIR
    )
