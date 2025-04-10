"""Feature engineering assets for external data using PyCaret with Dagster config system."""

from typing import Any

import dagster as dg
import polars as pl
from pycaret.clustering import ClusteringExperiment


class Defaults:
    """Default configuration values for feature engineering."""

    # Experiment settings
    SESSION_ID = 42

    # Normalization settings
    NORMALIZE = True
    NORM_METHOD = "robust"

    # Outlier detection settings
    OUTLIER_DETECTION = True
    OUTLIER_THRESHOLD = 0.05
    OUTLIER_METHOD = "iforest"

    # PCA settings
    PCA_ACTIVE = True
    PCA_COMPONENTS = 0.8
    PCA_METHOD = "linear"

    # Imputation settings
    IMPUTATION_TYPE = "simple"
    NUMERIC_IMPUTATION = "mean"
    CATEGORICAL_IMPUTATION = "mode"

    # Metadata settings
    METADATA_DETAIL = "full"

    # Clustering algorithm
    ALGORITHM = "kmeans"


@dg.asset(
    name="external_fe_raw_data",
    description="Loads raw external data for clustering",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["preprocessed_external_data"],
    required_resource_keys={"external_data_reader"},
)
def external_fe_raw_data(
    context: dg.AssetExecutionContext,
) -> dict[str, pl.DataFrame]:
    """Load raw external data using the configured reader resource.

    This asset depends on preprocessed_external_data to ensure the preprocessing
    pipeline completes before clustering starts.

    Args:
        context: Asset execution context with access to resources and logging

    Returns:
        Dictionary mapping category names to their respective dataframes
    """
    context.log.info("Loading external data by category")
    return context.resources.external_data_reader.read()


@dg.asset(
    name="external_filtered_features",
    description="Filters out ignored features from raw external data",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_fe_raw_data"],
    required_resource_keys={"config"},
)
def external_filtered_features(
    context: dg.AssetExecutionContext,
    external_fe_raw_data: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Filter out features that should be ignored using PyCaret's ignore_features.

    Uses the configuration to determine which features should be excluded from
    the analysis. Features listed in the ignore_features config will be removed
    from all category dataframes where they exist.

    Args:
        context: Asset execution context with access to resources and logging
        external_fe_raw_data: Dictionary of raw dataframes by category from external sources

    Returns:
        Dictionary of dataframes with specified features filtered out
    """
    filtered_data = {}

    # Get features to ignore from context resources
    ignore_features = getattr(context.resources.config, "ignore_features", [])

    if not ignore_features:
        context.log.info("No features to ignore, returning raw data")
        return external_fe_raw_data

    context.log.info(f"Features to ignore: {ignore_features}")

    # Storing original features for metadata
    original_features = {}

    for category, df in external_fe_raw_data.items():
        # Store original feature count
        original_features[category] = df.columns

        # Get features that actually exist in this dataframe
        features_to_ignore = [col for col in ignore_features if col in df.columns]

        if not features_to_ignore:
            context.log.info(f"No features to ignore for {category}")
            filtered_data[category] = df
            continue

        context.log.info(
            f"Ignoring features via PyCaret setup for {category}: {features_to_ignore}"
        )

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment with ignore_features
        exp = ClusteringExperiment()
        exp.setup(
            data=pandas_df,
            ignore_features=features_to_ignore,
            session_id=Defaults.SESSION_ID,
            verbose=False,
        )

        # Get the transformed data with ignored features removed
        filtered_data[category] = pl.from_pandas(exp.X_train_transformed)

        context.log.info(f"Removed features from {category}: {features_to_ignore}")

    # Store the ignored features in the context metadata for potential later use
    context.add_output_metadata(
        {"ignored_features": ignore_features, "original_features": original_features}
    )

    return filtered_data


@dg.asset(
    name="external_imputed_features",
    description="Imputes missing values in external features using PyCaret",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_filtered_features"],
    required_resource_keys={"config"},
)
def external_imputed_features(
    context: dg.AssetExecutionContext,
    external_filtered_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Impute missing values in external features using PyCaret's imputation methods.

    Handles missing values in each category dataframe using the configured
    imputation strategy. Imputation is configurable for both numeric and
    categorical features.

    Args:
        context: Asset execution context with access to resources and logging
        external_filtered_features: Dictionary of filtered dataframes by category

    Returns:
        Dictionary of dataframes with missing values imputed

    Notes:
        Configuration parameters:
        - imputation_type: Options are 'simple' or 'iterative'
        - numeric_imputation: Method for numeric features ('mean', 'median', etc.)
        - categorical_imputation: Method for categorical features ('mode', etc.)
    """
    processed_data = {}

    # Get all configuration parameters with defaults
    imputation_type = getattr(context.resources.config, "imputation_type", Defaults.IMPUTATION_TYPE)
    numeric_imputation = getattr(
        context.resources.config, "numeric_imputation", Defaults.NUMERIC_IMPUTATION
    )
    categorical_imputation = getattr(
        context.resources.config, "categorical_imputation", Defaults.CATEGORICAL_IMPUTATION
    )

    # Store original data for each category to potentially restore ignored features later
    original_data = {}

    for category, df in external_filtered_features.items():
        context.log.info(f"Imputing missing values for category: {category}")

        # Store original data
        original_data[category] = df

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment
        exp = ClusteringExperiment()

        setup_params = {
            "data": pandas_df,
            "imputation_type": imputation_type,
            "numeric_imputation": numeric_imputation,
            "categorical_imputation": categorical_imputation,
            "verbose": False,
            "session_id": Defaults.SESSION_ID,
        }

        exp.setup(**setup_params)

        # Get the imputed data
        processed_data[category] = pl.from_pandas(exp.X_train_transformed)

        context.log.info(f"Missing value imputation completed for {category}")

    # Store original data in context for potential restoration of ignored features
    context.add_output_metadata(
        {"original_data_shape": {k: v.shape for k, v in original_data.items()}}
    )

    return processed_data


@dg.asset(
    name="external_normalized_data",
    description="Applies feature scaling/normalization to external data using PyCaret",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_imputed_features"],
    required_resource_keys={"config"},
)
def external_normalized_data(
    context: dg.AssetExecutionContext,
    external_imputed_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Normalize external features using PyCaret's normalization methods.

    Applies feature scaling/normalization to numeric features based on the
    configured method. Normalization can be disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        external_imputed_features: Dictionary of imputed dataframes by category

    Returns:
        Dictionary of normalized dataframes by category
    """
    normalized_data = {}

    # Get settings from config
    normalize = getattr(context.resources.config, "normalize", Defaults.NORMALIZE)
    norm_method = getattr(context.resources.config, "norm_method", Defaults.NORM_METHOD)

    if not normalize:
        context.log.info("Normalization disabled, returning imputed data")
        return external_imputed_features

    context.log.info(f"Applying normalization with method: {norm_method}")

    for category, df in external_imputed_features.items():
        context.log.info(f"Normalizing features for category: {category}")

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment
        exp = ClusteringExperiment()

        # Set up experiment with normalization settings
        exp.setup(
            data=pandas_df,
            normalize=normalize,
            normalize_method=norm_method,
            verbose=False,
            session_id=Defaults.SESSION_ID,
        )

        # Get the normalized data
        normalized_data[category] = pl.from_pandas(exp.X_train_transformed)

        context.log.info(f"Normalization completed for {category}")

    # Add metadata
    context.add_output_metadata(
        {
            "normalization": {
                "enabled": normalize,
                "method": norm_method,
            }
        }
    )

    return normalized_data


@dg.asset(
    name="external_outlier_removed_features",
    description="Detects and removes outliers from external data",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_normalized_data"],
    required_resource_keys={"config"},
)
def external_outlier_removed_features(
    context: dg.AssetExecutionContext,
    external_normalized_data: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Detect and remove outliers from external data using PyCaret's outlier detection.

    Uses the configured outlier detection method to identify and remove
    outliers from the dataset. This step can be disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        external_normalized_data: Dictionary of normalized dataframes by category

    Returns:
        Dictionary of dataframes with outliers removed
    """
    outlier_free_data = {}

    # Get outlier detection settings
    outlier_detection = getattr(
        context.resources.config, "outlier_detection", Defaults.OUTLIER_DETECTION
    )
    outlier_method = getattr(context.resources.config, "outlier_method", Defaults.OUTLIER_METHOD)
    outlier_threshold = getattr(
        context.resources.config, "outlier_threshold", Defaults.OUTLIER_THRESHOLD
    )

    if not outlier_detection:
        context.log.info("Outlier detection disabled, returning normalized data")
        return external_normalized_data

    context.log.info(
        f"Performing outlier detection using {outlier_method} with threshold {outlier_threshold}"
    )

    outlier_summary = {}

    for category, df in external_normalized_data.items():
        context.log.info(f"Detecting outliers for category: {category}")

        # Skip datasets that are too small
        if len(df) < 10:
            context.log.warning(
                f"Category {category} has only {len(df)} samples, skipping outlier detection"
            )
            outlier_free_data[category] = df
            outlier_summary[category] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment
        exp = ClusteringExperiment()

        # Set up experiment with outlier detection settings
        exp.setup(
            data=pandas_df,
            remove_outliers=outlier_detection,
            outliers_method=outlier_method,
            outliers_threshold=outlier_threshold,
            verbose=False,
            session_id=Defaults.SESSION_ID,
        )

        # Get the data with outliers removed
        outlier_free_data[category] = pl.from_pandas(exp.X_train_transformed)

        # Calculate outlier metrics
        original_count = len(df)
        cleaned_count = len(outlier_free_data[category])
        outliers_removed = original_count - cleaned_count
        outlier_percentage = (outliers_removed / original_count) * 100 if original_count > 0 else 0

        context.log.info(
            f"Removed {outliers_removed} outliers ({outlier_percentage:.2f}%) from {category}"
        )

        # Store outlier metrics
        outlier_summary[category] = {
            "status": "success",
            "original_count": original_count,
            "final_count": cleaned_count,
            "outliers_removed": outliers_removed,
            "outlier_percentage": outlier_percentage,
        }

    # Add outlier summary to metadata
    context.add_output_metadata(
        {
            "outlier_detection": {
                "enabled": outlier_detection,
                "method": outlier_method,
                "threshold": outlier_threshold,
                "summary": outlier_summary,
            }
        }
    )

    return outlier_free_data


@dg.asset(
    name="external_dimensionality_reduced_features",
    description="Reduces external feature dimensions using PCA",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_outlier_removed_features"],
    required_resource_keys={"config"},
)
def external_dimensionality_reduced_features(
    context: dg.AssetExecutionContext,
    external_outlier_removed_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Reduce dimensionality of external features using PCA via PyCaret.

    Applies Principal Component Analysis to reduce the number of features
    while retaining most of the variance. This step can be disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        external_outlier_removed_features: Dictionary of dataframes with outliers removed

    Returns:
        Dictionary of dataframes with reduced dimensions
    """
    reduced_data = {}

    # Get PCA settings
    pca_active = getattr(context.resources.config, "pca_active", Defaults.PCA_ACTIVE)
    pca_components = getattr(context.resources.config, "pca_components", Defaults.PCA_COMPONENTS)
    pca_method = getattr(context.resources.config, "pca_method", Defaults.PCA_METHOD)

    if not pca_active:
        context.log.info("PCA disabled, returning data with outliers removed")
        return external_outlier_removed_features

    context.log.info(
        f"Performing dimensionality reduction using {pca_method} PCA "
        f"with components={pca_components}"
    )

    pca_summary = {}

    for category, df in external_outlier_removed_features.items():
        context.log.info(f"Reducing dimensions for category: {category}")

        # Skip datasets with too few samples or features
        if len(df) < 3 or df.width < 3:
            context.log.warning(
                f"Category {category} has only {len(df)} samples and {df.width} features, "
                f"skipping dimensionality reduction"
            )
            reduced_data[category] = df
            pca_summary[category] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize PyCaret experiment
        exp = ClusteringExperiment()

        # Set up experiment with PCA settings
        exp.setup(
            data=pandas_df,
            pca=pca_active,
            pca_method=pca_method,
            pca_components=pca_components,
            verbose=False,
            session_id=Defaults.SESSION_ID,
        )

        # Get the data with reduced dimensions
        reduced_data[category] = pl.from_pandas(exp.X_train_transformed)

        # Calculate PCA metrics
        original_features = df.width
        reduced_features = reduced_data[category].width
        reduction_percentage = (
            ((original_features - reduced_features) / original_features) * 100
            if original_features > 0
            else 0
        )

        context.log.info(
            f"Reduced {category} from {original_features} to {reduced_features} features "
            f"({reduction_percentage:.2f}% reduction)"
        )

        # Store PCA metrics
        pca_summary[category] = {
            "status": "success",
            "original_features": original_features,
            "reduced_features": reduced_features,
            "reduction_percentage": reduction_percentage,
        }

    # Add PCA summary to metadata
    context.add_output_metadata(
        {
            "pca": {
                "enabled": pca_active,
                "method": pca_method,
                "components": pca_components,
                "summary": pca_summary,
            }
        }
    )

    return reduced_data


@dg.asset(
    name="external_feature_metadata",
    description="External feature engineering metadata asset",
    group_name="clustering",
    compute_kind="external_clustering_feature_engineering",
    deps=["external_dimensionality_reduced_features"],
    required_resource_keys={"config"},
)
def external_feature_metadata(
    context: dg.AssetExecutionContext,
    external_dimensionality_reduced_features: dict[str, pl.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Generate metadata about the external feature engineering process.

    Creates a metadata asset that summarizes the feature engineering steps
    and the resulting data characteristics.

    Args:
        context: Asset execution context with access to resources and logging
        external_dimensionality_reduced_features: Dictionary of dataframes with reduced dimensions

    Returns:
        Dictionary containing metadata about the feature engineering process
    """
    metadata_detail = getattr(context.resources.config, "metadata_detail", Defaults.METADATA_DETAIL)
    metadata = {}

    context.log.info(f"Generating feature metadata with detail level: {metadata_detail}")

    # Gather basic metrics for each category
    for category, df in external_dimensionality_reduced_features.items():
        category_metadata = {
            "record_count": len(df),
            "feature_count": df.width,
            "feature_names": df.columns,
            "non_null_counts": {
                col: df[col].filter(pl.col(col).is_not_null()).height for col in df.columns
            },
            "data_types": {col: str(df[col].dtype) for col in df.columns},
        }

        # Add advanced statistical metrics if full detail requested
        if metadata_detail == "full":
            try:
                # Add numeric column statistics
                numeric_cols = [
                    col
                    for col in df.columns
                    if df[col].dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Float32)
                ]

                if numeric_cols:
                    stats = df.select(
                        [pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols]
                        + [pl.col(col).std().alias(f"{col}_std") for col in numeric_cols]
                        + [pl.col(col).min().alias(f"{col}_min") for col in numeric_cols]
                        + [pl.col(col).max().alias(f"{col}_max") for col in numeric_cols]
                    )

                    stats_dict = stats.row(0)

                    category_metadata["statistics"] = {
                        col: {
                            "mean": stats_dict.get(f"{col}_mean"),
                            "std": stats_dict.get(f"{col}_std"),
                            "min": stats_dict.get(f"{col}_min"),
                            "max": stats_dict.get(f"{col}_max"),
                        }
                        for col in numeric_cols
                    }
            except Exception as e:
                context.log.error(f"Error calculating statistics for {category}: {str(e)}")
                category_metadata["statistics_error"] = str(e)

        metadata[category] = category_metadata

    # Add metadata to the context
    context.add_output_metadata(
        {
            "detail_level": metadata_detail,
            "categories": list(metadata.keys()),
            "total_record_count": sum(meta["record_count"] for meta in metadata.values()),
        }
    )

    return metadata
