"""
pipeline_utils.py

Utility functions and classes for data preprocessing, feature transformation,
PCA reduction, clustering, model training, and model explainability.
"""

import logging

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def preprocess_data(df, cat_onehot, cat_ordinal, num):
    """
    Preprocess the data using OneHotEncoder for categorical (if any),
    OrdinalEncoder for ordinal (if any), and StandardScaler for numeric columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cat_onehot (list): Columns to be one-hot encoded.
        cat_ordinal (list): Columns to be ordinal-encoded.
        num (list): Numeric columns to be standardized.

    Returns:
        data (pd.DataFrame): The transformed DataFrame.
        pipe_fit (Pipeline): The fitted pipeline object.
    """
    categorical_transformer_onehot = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
        ]
    )

    categorical_transformer_ordinal = Pipeline(steps=[("encoder", OrdinalEncoder())])

    num_transformer = Pipeline(steps=[("encoder", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_onehot", categorical_transformer_onehot, cat_onehot),
            ("cat_ordinal", categorical_transformer_ordinal, cat_ordinal),
            ("num", num_transformer, num),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit the pipeline on the full dataset
    pipe_fit = pipeline.fit(df)

    # Transform data and reassemble into a DataFrame
    data = pd.DataFrame(pipe_fit.transform(df), columns=pipe_fit.get_feature_names_out().tolist())

    return data, pipe_fit


def drop_correlated_features(data, corr_threshold=0.80, features_to_keep=None):
    """
    Checks the correlation of numeric features and drops features that exceed the given threshold.

    For each pair of highly correlated columns (over corr_threshold), we normally keep the first one
    and drop the second. However, any feature in 'features_to_keep' is never dropped.

    Parameters:
        data (pd.DataFrame): The DataFrame with numeric (and possibly non-numeric) features.
        corr_threshold (float): The correlation threshold above which features are dropped.
        features_to_keep (list): A list of feature names that must *not* be dropped, even if
                                 they are highly correlated.

    Returns:
        reduced_data (pd.DataFrame): The DataFrame after dropping correlated features,
                                     but keeping those in 'features_to_keep'.
    """
    if features_to_keep is None:
        features_to_keep = []

    # We only evaluate numeric columns for correlation
    numeric_df = data.select_dtypes(include=[np.number])
    initial_features = numeric_df.columns.tolist()
    logging.info(f"Initial number of numeric features: {len(initial_features)}")

    # Compute absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Identify all pairs that exceed the correlation threshold
    high_corr_pairs = [
        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iat[i, j])
        for i in range(len(corr_matrix.columns))
        for j in range(i + 1, len(corr_matrix.columns))
        if corr_matrix.iat[i, j] > corr_threshold
    ]

    columns_to_drop = set()
    dropped_in_favor_of = {}

    # Process each high-correlation pair
    for col1, col2, corr_value in high_corr_pairs:
        # If either col1 or col2 was already dropped, skip
        if col1 in columns_to_drop or col2 in columns_to_drop:
            continue

        # If both are in features_to_keep, keep both
        if (col1 in features_to_keep) and (col2 in features_to_keep):
            continue

        # If col1 is in features_to_keep but col2 is not, drop col2
        if (col1 in features_to_keep) and (col2 not in features_to_keep):
            columns_to_drop.add(col2)
            dropped_in_favor_of[col2] = (col1, corr_value)
            continue

        # If col2 is in features_to_keep but col1 is not, drop col1
        if (col2 in features_to_keep) and (col1 not in features_to_keep):
            columns_to_drop.add(col1)
            dropped_in_favor_of[col1] = (col2, corr_value)
            continue

        # Otherwise, neither col1 nor col2 is in the must-keep list,
        # so we follow the original logic: keep the first, drop the second.
        columns_to_drop.add(col2)
        dropped_in_favor_of[col2] = (col1, corr_value)

    logging.info(f"Number of features to drop: {len(columns_to_drop)}")

    # Log each dropped feature and the feature that was kept
    for dropped_feature, (kept_feature, corr_val) in dropped_in_favor_of.items():
        logging.info(
            f"Feature '{dropped_feature}' was dropped due to high correlation ({corr_val:.3f}) "
            f"with '{kept_feature}' which is kept."
        )

    reduced_data = data.drop(columns=columns_to_drop, errors="ignore")
    logging.info(f"Number of features after drop: {reduced_data.shape[1]}")

    return reduced_data


def apply_pca(
    data,
    pca_n_features_threshold,
    max_n_pca_features,
    cumulative_variance_target=0.95,
    random_state=9999,
):
    """
    Applies PCA to the data if the number of features exceeds pca_n_features_threshold.
    Reduces the number of components to either reach the cumulative_variance_target or max_n_pca_features,
    whichever occurs first.

    Parameters:
        data (pd.DataFrame): Numeric data to be reduced.
        pca_n_features_threshold (int): Minimum number of features above which PCA is applied.
        max_n_pca_features (int): Maximum PCA components to keep.
        cumulative_variance_target (float): Target cumulative variance ratio.
        random_state (int): Random state for reproducibility.

    Returns:
        transformed_data (pd.DataFrame): Data after PCA (if applied) or the original data if not.
        pca (PCA or None): The fitted PCA object if PCA was applied, otherwise None.
    """
    n_features = data.shape[1]
    if n_features > pca_n_features_threshold:
        pca = PCA(random_state=random_state)
        pca.fit(data)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_needed = np.searchsorted(cumulative_variance, cumulative_variance_target) + 1
        n_components_final = min(n_components_needed, max_n_pca_features)

        logging.info(
            f"Applying PCA: initial features={n_features}, "
            f"target variance={cumulative_variance_target}, "
            f"components after PCA={n_components_final}"
        )

        pca = PCA(n_components=n_components_final, random_state=random_state)
        transformed_data = pca.fit_transform(data)

        # Construct a new DataFrame with PCA component names
        component_names = [f"PC{i}" for i in range(1, n_components_final + 1)]
        transformed_data = pd.DataFrame(transformed_data, columns=component_names)
        return transformed_data, pca
    else:
        logging.info("PCA not applied because number of features <= pca_n_features_threshold.")
        return data, None


def run_kmeans_and_evaluate(
    data,
    min_n_clusters_to_try,
    max_n_clusters_to_try,
    small_cluster_threshold,
    random_state=9999,
    max_iter=500,
):
    """
    Runs K-Means for a range of cluster values, checking multiple metrics
    (Silhouette, Calinski-Harabasz, Davies-Bouldin).
    Filters out runs that generate any small clusters below small_cluster_threshold.

    Parameters:
        data (pd.DataFrame or np.ndarray): The data to cluster.
        min_n_clusters_to_try (int): Minimum number of clusters to try.
        max_n_clusters_to_try (int): Maximum number of clusters to try.
        small_cluster_threshold (int): Minimum allowed cluster size.
        random_state (int): Random state for reproducibility.
        max_iter (int): Maximum number of iterations for K-Means.

    Returns:
        results_df (pd.DataFrame): DataFrame of results with metrics, normalized metrics, combined score.
        best_k (int or None): Best cluster count based on highest combined score; None if none valid.
    """
    results = []
    X = data.values if hasattr(data, "values") else data
    run_number = 0

    # Store raw results first
    raw_results = []
    for k in range(min_n_clusters_to_try, max_n_clusters_to_try + 1):
        run_number += 1
        km = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init="auto")
        labels = km.fit_predict(X)

        # Check cluster sizes
        cluster_sizes = np.bincount(labels)

        # Only keep runs that meet the small_cluster_threshold
        if all(size >= small_cluster_threshold for size in cluster_sizes):
            sil = silhouette_score(X, labels) if k > 1 else np.nan
            chs = calinski_harabasz_score(X, labels)
            dbs = davies_bouldin_score(X, labels)

            raw_results.append(
                {
                    "run_number": run_number,
                    "n_clusters": k,
                    "silhouette_score": sil,
                    "calinski_harabasz_score": chs,
                    "davies_bouldin_score": dbs,
                }
            )

    # If no valid runs remain, return an empty DataFrame
    if not raw_results:
        logging.warning("No valid runs met the small_cluster_threshold.")
        return pd.DataFrame(), None

    results_df = pd.DataFrame(raw_results)

    # Normalize metrics to create a combined score
    eps = 1e-10

    S_min, S_max = results_df["silhouette_score"].min(), results_df["silhouette_score"].max()
    C_min, C_max = (
        results_df["calinski_harabasz_score"].min(),
        results_df["calinski_harabasz_score"].max(),
    )
    D_min, D_max = (
        results_df["davies_bouldin_score"].min(),
        results_df["davies_bouldin_score"].max(),
    )

    # Normalized silhouette
    results_df["silhouette_norm"] = (results_df["silhouette_score"] - S_min) / (S_max - S_min + eps)
    # Normalized CH
    results_df["chs_norm"] = (results_df["calinski_harabasz_score"] - C_min) / (C_max - C_min + eps)
    # Invert Davies-Bouldin (lower is better, so let's invert it)
    results_df["dbs_inv"] = (D_max - results_df["davies_bouldin_score"]) / (D_max - D_min + eps)

    # Combined metric as an average of the three normalized scores
    results_df["combined_score"] = (
        results_df["silhouette_norm"] + results_df["chs_norm"] + results_df["dbs_inv"]
    ) / 3.0

    # Identify the best run
    best_run = results_df.loc[results_df["combined_score"].idxmax()]
    best_k = int(best_run["n_clusters"])

    logging.info("Optimal run based on combined metric:")
    logging.info(best_run)

    return results_df, best_k


def add_cluster_labels(df, pipeline, n_clusters, random_state=9999, max_iter=500):
    """
    Adds cluster labels to the original dataframe using K-Means on the preprocessed data.

    Parameters:
        df (pd.DataFrame): The original DataFrame before clustering.
        pipeline (Pipeline): A fitted preprocessing pipeline to transform df before clustering.
        n_clusters (int): The number of clusters to fit with K-Means.
        random_state (int): Random state for reproducibility.
        max_iter (int): Maximum iterations for K-Means.

    Returns:
        df_with_clusters (pd.DataFrame): Original DataFrame with 'cluster_label' assigned to each row.
    """
    # Transform the original dataframe using the fitted pipeline
    data_transformed = pipeline.transform(df)

    # Run K-Means on the transformed data
    km = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init="auto")
    labels = km.fit_predict(data_transformed)

    # Copy original df and append the cluster labels
    df_with_clusters = df.copy()
    df_with_clusters["cluster_label"] = labels

    return df_with_clusters


def plot_metrics_and_elbow(
    data, results_df, min_n_clusters_to_try, max_n_clusters_to_try, random_state=9999, max_iter=500
):
    """
    Generates four plots:
      1. Silhouette score vs number of clusters
      2. Calinski-Harabasz score vs number of clusters
      3. Davies-Bouldin score vs number of clusters
      4. Elbow Method (Inertia vs number of clusters)

    Highlights the "optimal" number of clusters (based on highest combined score).

    Parameters:
        data (pd.DataFrame or np.ndarray): The dataset used for clustering.
        results_df (pd.DataFrame): Contains scores for each valid run.
        min_n_clusters_to_try (int): Minimum number of clusters tested.
        max_n_clusters_to_try (int): Maximum number of clusters tested.
        random_state (int): Random state for reproducibility.
        max_iter (int): Maximum iterations for K-Means.
    """
    if results_df.empty:
        logging.warning("results_df is empty. No plots will be generated.")
        return

    # Identify the best run by combined_score
    best_run = results_df.loc[results_df["combined_score"].idxmax()]
    best_k = best_run["n_clusters"]

    n_clusters = results_df["n_clusters"]
    silhouette_scores = results_df["silhouette_score"]
    ch_scores = results_df["calinski_harabasz_score"]
    db_scores = results_df["davies_bouldin_score"]

    # Use discrete ticks for cluster counts that actually ran successfully
    cluster_ticks = sorted(n_clusters.unique())

    # 1. Silhouette Score Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_clusters, silhouette_scores, marker="o", label="Silhouette Score")
    ax.axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={int(best_k)}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs Number of Clusters")
    ax.set_xticks(cluster_ticks)
    ax.set_xticklabels([str(int(k)) for k in cluster_ticks])
    ax.legend()
    plt.show()

    # 2. Calinski-Harabasz Score Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_clusters, ch_scores, marker="o", label="Calinski-Harabasz Score")
    ax.axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={int(best_k)}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Calinski-Harabasz Score")
    ax.set_title("Calinski-Harabasz Score vs Number of Clusters")
    ax.set_xticks(cluster_ticks)
    ax.set_xticklabels([str(int(k)) for k in cluster_ticks])
    ax.legend()
    plt.show()

    # 3. Davies-Bouldin Score Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_clusters, db_scores, marker="o", label="Davies-Bouldin Score")
    ax.axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={int(best_k)}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Davies-Bouldin Score")
    ax.set_title("Davies-Bouldin Score vs Number of Clusters")
    ax.set_xticks(cluster_ticks)
    ax.set_xticklabels([str(int(k)) for k in cluster_ticks])
    ax.legend()
    plt.show()

    # 4. Elbow Method (Inertia vs Number of Clusters)
    inertias = []
    cluster_range = range(min_n_clusters_to_try, max_n_clusters_to_try + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init="auto")
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cluster_range, inertias, marker="o", label="Inertia")
    ax.axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={int(best_k)}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method: Inertia vs Number of Clusters")
    ax.set_xticks(cluster_ticks)
    ax.set_xticklabels([str(int(k)) for k in cluster_ticks])
    ax.legend()
    plt.show()


def train_and_explain_model(df, target_col, drop_col="STORE_NBR"):
    """
    Train a LightGBM model, explain it with SHAP, and print accuracy & classification report.

    Parameters:
        df (pd.DataFrame): The input DataFrame for training.
        target_col (str): Name of the target column in df.
        drop_col (str): Name of any column to drop from df features (defaults to 'STORE_NBR').

    Returns:
        None
    """
    # Prepare feature matrix and target
    X_train = df.drop(columns=[drop_col, target_col])
    y_train = df[target_col]

    # Initialize and train LightGBM
    clf_km = lgb.LGBMClassifier(colsample_by_tree=0.8)
    clf_km.fit(X=X_train, y=y_train, feature_name="auto", categorical_feature="auto")

    # Generate SHAP explanations
    explainer_km = shap.TreeExplainer(clf_km)
    shap_values_km = explainer_km.shap_values(X_train)

    # SHAP summary plot (bar)
    shap.summary_plot(shap_values_km, X_train, plot_type="bar", plot_size=(15, 5))

    # Predictions and evaluation
    y_pred = clf_km.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Training-set accuracy score: {accuracy:.4f}")
    print(classification_report(y_train, y_pred))
