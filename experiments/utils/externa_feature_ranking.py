import logging
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb
import shap

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# from configs.config import (
#     STORE_COL, TARGET_COL, COLUMNS_TO_IGNORE,
#     TEST_SIZE, RANDOM_STATE,
#     NUM_LEAVES, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, STOPPING_ROUNDS
# )


# ----------------------------------------------
# 1) Data Preparation Function
# ----------------------------------------------
def prepare_data_for_model(
    model_df: pd.DataFrame,
    store_col: str,
    target_col: str,
    columns_to_ignore: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prepare the data for multiclass classification:
      1. Set 'store_col' as index (or remove it) so it's not a feature.
      2. Convert the 'target_col' to string (e.g., for multi-class).
      3. Remove 'columns_to_ignore' from predictors (if provided).
      4. Perform a stratified train_test_split.

    Parameters
    ----------
    model_df : pd.DataFrame
        The DataFrame containing all features + the cluster_label.
    store_col : str
        The column name for the store identifier (e.g., 'STORE_NBR').
    target_col : str
        The column name for the multiclass target (e.g., 'cluster_label').
    columns_to_ignore : list
        Columns that should NOT be used as predictors.
    test_size : float
        Fraction of data to use for test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target (string).
    y_test : pd.Series
        Test target (string).
    feature_cols : list
        The list of feature column names actually used for modeling.
    """
    logger.info("Starting data preparation for model.")

    if columns_to_ignore is None:
        columns_to_ignore = []

    # Make a copy to avoid mutating the original DataFrame
    df = model_df.copy()

    # 1) Drop or set store_col as the index (to remove it from features)
    if store_col in df.columns:
        logger.debug(f"Setting '{store_col}' as DataFrame index and removing it from features.")
        df.set_index(store_col, inplace=True, drop=True)

    # 2) Convert target to string
    logger.debug(f"Converting '{target_col}' to string data type.")
    df[target_col] = df[target_col].astype(str)

    # 3) Determine which columns to exclude from features
    exclude_cols = columns_to_ignore + [target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 4) Create X (features) and y (target)
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 5) Perform stratified train/test split
    logger.info(f"Performing train_test_split: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Data preparation complete.")
    logger.debug(f"X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


# ----------------------------------------------
# 2) Training Function
# ----------------------------------------------
def train_lgbm_multiclass(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list,
    num_leaves: int = 31,
    max_depth: int = -1,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    stopping_rounds: int = 10,
):
    """
    Train a LightGBM multi-class classifier and evaluate on the test data.

    Returns
    -------
    model : lgb.LGBMClassifier
        The trained LightGBM model.
    top_features_df : pd.DataFrame
        Feature names & importances, sorted descending.
    acc : float
        Accuracy on the test set.
    class_report_str : str
        The classification report as a string.
    conf_matrix_str : str
        The confusion matrix as a string representation.
    """
    logger.info("Initializing LightGBM multi-class classifier.")

    # Create the LightGBM classifier
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_leaves=num_leaves,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    # Prepare callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds),
        lgb.log_evaluation(period=0),  # Suppress per-iteration logging
    ]

    logger.info("Fitting LightGBM model with early stopping.")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )

    logger.info("Model training complete. Generating predictions.")
    y_pred = model.predict(X_test)

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")

    # Convert classification_report to string
    class_report_str = classification_report(y_test, y_pred)
    logger.info("\nClassification Report:\n" + class_report_str)

    # Convert confusion_matrix to a string
    cm = confusion_matrix(y_test, y_pred)
    cm_str = str(cm)
    logger.debug(f"Confusion Matrix:\n{cm_str}")

    # Build feature importances
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp_df.sort_values("importance", ascending=False, inplace=True)
    top_features_df = feat_imp_df.reset_index(drop=True)

    logger.info("LightGBM training and evaluation complete.")
    return model, top_features_df, acc, class_report_str, cm_str


# ----------------------------------------------
# 3) Compute SHAP Values (Global) Function
# ----------------------------------------------
def compute_shap_values(model, X_test: pd.DataFrame, top_n: int = None):
    """
    Compute global SHAP values for a LightGBM model (multi-class or binary).
    Aggregates feature contributions across classes if multi-class.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained LightGBM model.
    X_test : pd.DataFrame
        The data on which to compute SHAP values.
    top_n : int
        If not None, return only the top N features by mean absolute SHAP.

    Returns
    -------
    shap_df : pd.DataFrame
        Columns: [feature, mean_abs_shap, directionality]
        - 'mean_abs_shap': average magnitude of SHAP across classes + samples
        - 'directionality': overall sign (positive/negative)
    """
    logger.info("Computing global SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Check SHAP output shape or type
    if isinstance(shap_values, list):
        logger.debug(
            f"Detected list of SHAP arrays (multi-class). List length = {len(shap_values)}"
        )
    else:
        logger.debug(f"Detected SHAP array with shape: {shap_values.shape}")

    # CASE A: multi-class => shap_values is a list of arrays or a 3D array
    if isinstance(shap_values, list):
        # Some SHAP versions include an extra baseline array => drop it if length = n_classes+1
        n_classes = getattr(model, "n_classes_", len(shap_values))
        if len(shap_values) == n_classes + 1:
            shap_values = shap_values[:-1]
        shap_array = np.array(shap_values)  # shape => (n_classes, n_samples, n_features)

        # Compute mean_abs_shap by taking mean across samples for each class, then summing across classes
        shap_abs = np.abs(shap_array).mean(axis=1)  # => (n_classes, n_features)
        mean_abs_shap_values = shap_abs.sum(axis=0)  # => (n_features,)

        # Directionality: average the signed SHAP across classes -> (n_samples, n_features)
        avg_shap_across_classes = shap_array.mean(axis=0)
        sum_across_classes = avg_shap_across_classes.sum(axis=0)  # => (n_features,)
        directionality = np.where(sum_across_classes >= 0, "positive", "negative")

    else:
        # CASE B: binary or single-array multi-class => shape could be (n_samples, n_features) or (n_samples, n_features, n_classes)
        shap_array = shap_values
        if shap_array.ndim == 2:
            # Binary or single-class
            mean_abs_shap_values = np.abs(shap_array).mean(axis=0)
            sum_over_samples = shap_array.sum(axis=0)
            directionality = np.where(sum_over_samples >= 0, "positive", "negative")
        elif shap_array.ndim == 3:
            # Multi-class => (n_samples, n_features, n_classes)
            shap_abs = np.abs(shap_array).mean(axis=0)  # => (n_features, n_classes)
            mean_abs_shap_values = shap_abs.sum(axis=1)  # => (n_features,)
            avg_shap_across_samples = shap_array.mean(axis=0)
            sum_across_classes = avg_shap_across_samples.sum(axis=1)
            directionality = np.where(sum_across_classes >= 0, "positive", "negative")
        else:
            raise ValueError("Unexpected SHAP array shape. Expected 2D or 3D.")

    if len(mean_abs_shap_values) != X_test.shape[1]:
        raise ValueError(
            "Mismatch between # of features in SHAP output and X_test columns. "
            "Verify preprocessing alignment."
        )

    # Build final DataFrame
    shap_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap_values,
            "directionality": directionality,
        }
    )

    shap_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    if top_n is not None:
        shap_df = shap_df.head(top_n).reset_index(drop=True)

    logger.info("Global SHAP values computed successfully.")
    return shap_df


# ----------------------------------------------
# 4) Compute SHAP Values (Per-Class) Function
# ----------------------------------------------
def compute_shap_values_per_class(model, X_test: pd.DataFrame):
    """
    Compute per-class SHAP contributions for a LightGBM model (multi-class).
    Returns a "long" DataFrame with one row per (feature, class).

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained LightGBM model.
    X_test : pd.DataFrame
        The data to compute SHAP values for.

    Returns
    -------
    shap_df_long : pd.DataFrame
        Columns: [feature, cluster_label, avg_shap_value, directionality, mean_abs_shap]
        Each row = (feature, one of the classes).
    """
    logger.info("Computing per-class SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Convert shap_values to a standard 3D array => (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        # If there's an extra baseline array, drop it
        if hasattr(model, "n_classes_") and len(shap_values) == model.n_classes_ + 1:
            shap_values = shap_values[:-1]
            n_classes = len(shap_values)
        shap_array = np.stack(shap_values, axis=-1)  # => (n_samples, n_features, n_classes)
    else:
        shap_array = shap_values
        if shap_array.ndim == 2:
            # Single-class or binary => make it 3D
            shap_array = shap_array[:, :, np.newaxis]
        n_classes = shap_array.shape[2]

    # Build a "long" data table
    df_list = []
    for c in range(n_classes):
        shap_c = shap_array[:, :, c]  # => (n_samples, n_features)
        avg_shap_c = shap_c.mean(axis=0)
        directionality_c = np.where(avg_shap_c >= 0, "positive", "negative")
        mean_abs_shap_c = np.abs(shap_c).mean(axis=0)

        temp_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "cluster_label": str(c),
                "avg_shap_value": avg_shap_c,
                "directionality": directionality_c,
                "mean_abs_shap": mean_abs_shap_c,
            }
        )
        df_list.append(temp_df)

    shap_df_long = pd.concat(df_list, axis=0, ignore_index=True)
    logger.info("Per-class SHAP values computed successfully.")
    return shap_df_long


import pandas as pd
import re


def standardize_text(s: str) -> str:
    """
    Convert to uppercase, replace special characters with spaces,
    collapse extra spaces, and strip whitespace.
    """
    if pd.isnull(s):
        return s
    s = s.upper()
    # Replace `_`, `&`, `/`, and `-` with a space
    s = re.sub(r"[_&/\-]", " ", s)
    # Collapse multiple spaces -> single
    s = re.sub(r"\s+", " ", s)
    # Strip leading/trailing spaces
    return s.strip()


import pandas as pd


def get_top_80pct_features(grp):
    """
    For a single group of rows (specific REPO EXTERNAL CLUSTERING GROUP & CATEGORY),
    sort features by descending normalized_mean_abs_shap, then keep adding them
    until we reach exactly 0.8 total. If adding the next feature crosses 0.8,
    include only the partial contribution needed to get to 0.8.
    """
    grp = grp.sort_values("normalized_mean_abs_shap", ascending=False)
    csum = 0.0
    kept_rows = []

    for idx, row in grp.iterrows():
        value = row["normalized_mean_abs_shap"]
        if csum + value < 0.8:
            # We can safely add the entire feature
            csum += value
            kept_rows.append(row)
        else:
            # We only need a part of this feature to reach 0.8
            partial_needed = 0.8 - csum
            if partial_needed > 0:
                row_copy = row.copy()
                row_copy["normalized_mean_abs_shap"] = partial_needed
                kept_rows.append(row_copy)
                csum = 0.8
            break  # we've reached 0.8, so stop

    return pd.DataFrame(kept_rows)


def get_latest_internal_run_directory(internal_content_dir: str) -> str:
    """
    Scans the given directory (internal_content_dir) for subdirectories that
    start with 'Internal_Clustering_Run_' and returns the most recently
    named one by lexicographic sorting (which aligns with timestamps in the name).
    """
    subdirs = [
        d
        for d in os.listdir(internal_content_dir)
        if os.path.isdir(os.path.join(internal_content_dir, d))
        and d.startswith("Internal_Clustering_Run_")
    ]
    if not subdirs:
        raise ValueError(
            f"No subdirectories found matching 'Internal_Clustering_Run_' in {internal_content_dir}"
        )

    # Sort by name (which includes YYYYMMDD_HHMM)
    subdirs.sort()
    latest_subdir = subdirs[-1]
    return os.path.join(internal_content_dir, latest_subdir)


# ==============================================================================
# 1) FUNCTION TO GET LATEST GROUPED SHAP FILE FROM feature_ranking_results
# ==============================================================================
def get_latest_grouped_shap_file(feature_ranking_results_dir: str) -> str:
    """
    Searches within 'feature_ranking_results_dir' for subdirectories
    named Ranking_Run_{Date}_{Time}, picks the most recent one, and returns
    the path to the first .xlsx file that begins with 'grouped_shap_all_'.

    :param feature_ranking_results_dir: Base directory containing Ranking_Run_* subdirs.
    :return: Full path to the latest grouped_shap_all_ file.
    :raises FileNotFoundError: If no valid Ranking_Run_ directory or .xlsx file is found.
    """
    if not os.path.isdir(feature_ranking_results_dir):
        raise FileNotFoundError(
            f"Specified feature_ranking_results_dir does not exist: {feature_ranking_results_dir}"
        )

    # List subdirectories that start with 'Ranking_Run_'
    ranking_run_dirs = [
        d
        for d in os.listdir(feature_ranking_results_dir)
        if os.path.isdir(os.path.join(feature_ranking_results_dir, d))
        and d.startswith("Ranking_Run_")
    ]

    if not ranking_run_dirs:
        raise FileNotFoundError(
            f"No subdirectories starting with 'Ranking_Run_' found in {feature_ranking_results_dir}"
        )

    # Sort them by name (assuming they follow Ranking_Run_YYYYMMDD_HHMM format).
    # Lexicographical sorting is sufficient because YYYYMMDD_HHMM is comparable as a string.
    ranking_run_dirs.sort()

    # The "latest" one will be the last after sorting
    latest_dir_name = ranking_run_dirs[-1]
    latest_dir_path = os.path.join(feature_ranking_results_dir, latest_dir_name)

    # Find an .xlsx file that starts with grouped_shap_all_
    shap_files = [
        f
        for f in os.listdir(latest_dir_path)
        if f.startswith("grouped_shap_all_") and f.endswith(".xlsx")
    ]

    if not shap_files:
        raise FileNotFoundError(
            f"No file starting with 'grouped_shap_all_' found in {latest_dir_path}"
        )

    # If multiple, just pick the first. Adjust if you'd like further sorting/logic.
    chosen_file = shap_files[0]
    return os.path.join(latest_dir_path, chosen_file)
