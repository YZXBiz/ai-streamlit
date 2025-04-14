"""Feature engineering assets using PyCaret with Dagster config system."""

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
    name="internal_fe_raw_data",
    description="Loads raw sales data from pickle file",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_output_sales_table"],
    required_resource_keys={"output_sales_reader"},
)
def internal_fe_raw_data(
    context: dg.AssetExecutionContext,
) -> dict[str, pl.DataFrame]:
    """Load raw sales data from a file using the configured reader resource.

    This asset depends on internal_output_sales_table to ensure the preprocessing
    pipeline completes before feature engineering starts.

    Args:
        context: Asset execution context with access to resources and logging

    Returns:
        Dictionary mapping category names to their respective dataframes
    """
    context.log.info("Loading sales data by category")
    return context.resources.output_sales_reader.read()


@dg.asset(
    name="internal_filtered_features",
    description="Filters out ignored features from raw data",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_fe_raw_data"],
    required_resource_keys={"config"},
)
def internal_filtered_features(
    context: dg.AssetExecutionContext,
    internal_fe_raw_data: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Filter out features that should be ignored using PyCaret's ignore_features.

    Uses the configuration to determine which features should be excluded from
    the analysis. Features listed in the ignore_features config will be removed
    from all category dataframes where they exist.

    Args:
        context: Asset execution context with access to resources and logging
        internal_fe_raw_data: Dictionary of raw dataframes by category

    Returns:
        Dictionary of dataframes with specified features filtered out
    """
    filtered_data = {}

    # Get features to ignore from context resources
    ignore_features = getattr(context.resources.config, "ignore_features", [])

    if not ignore_features:
        context.log.info("No features to ignore, returning raw data")
        return internal_fe_raw_data

    context.log.info(f"Features to ignore: {ignore_features}")

    # Storing original features for metadata
    original_features = {}

    for category, df in internal_fe_raw_data.items():
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
    name="internal_imputed_features",
    description="Imputes missing values in features using PyCaret",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_filtered_features"],
    required_resource_keys={"config"},
)
def internal_imputed_features(
    context: dg.AssetExecutionContext,
    internal_filtered_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Impute missing values in features using PyCaret's imputation methods.

    Handles missing values in each category dataframe using the configured
    imputation strategy. Imputation is configurable for both numeric and
    categorical features.

    Args:
        context: Asset execution context with access to resources and logging
        internal_filtered_features: Dictionary of filtered dataframes by category

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

    for category, df in internal_filtered_features.items():
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
    name="internal_normalized_data",
    description="Applies feature scaling/normalization using PyCaret",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_imputed_features"],
    required_resource_keys={"config"},
)
def internal_normalized_data(
    context: dg.AssetExecutionContext,
    internal_imputed_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Normalize features using PyCaret's normalization methods.

    Applies feature scaling/normalization to numeric features based on the
    configured method. Normalization can be disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        internal_imputed_features: Dictionary of imputed dataframes by category

    Returns:
        Dictionary of dataframes with normalized features

    Notes:
        Configuration parameters:
        - normalize: Boolean to enable/disable normalization
        - norm_method: Method to use for normalization ('robust', 'zscore', etc.)
    """
    processed_data = {}

    # Check if normalization is enabled
    normalize = getattr(context.resources.config, "normalize", Defaults.NORMALIZE)
    if not normalize:
        context.log.info("Normalization disabled, returning imputed features")
        return internal_imputed_features

    # Get normalization method
    norm_method = getattr(context.resources.config, "norm_method", Defaults.NORM_METHOD)

    for category, df in internal_imputed_features.items():
        context.log.info(f"Normalizing features for category: {category}")

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize experiment
        exp = ClusteringExperiment()

        setup_params = {
            "data": pandas_df,
            "normalize": True,
            "normalize_method": norm_method,
            "verbose": False,
            "session_id": Defaults.SESSION_ID,
        }

        exp.setup(**setup_params)

        processed_data[category] = pl.from_pandas(exp.X_train_transformed)

        context.log.info(f"Normalization completed for {category}")

    return processed_data


@dg.asset(
    name="internal_outlier_removed_features",
    description="Detects and removes outliers",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_normalized_data"],
    required_resource_keys={"config"},
)
def internal_outlier_removed_features(
    context: dg.AssetExecutionContext,
    internal_normalized_data: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Detect and remove outliers using isolation forest or other methods.

    Applies outlier detection and removal using PyCaret. It processes
    each category-specific dataframe by detecting and removing outliers according
    to the configured method and threshold.

    Args:
        context: Asset execution context with access to resources and logging
        internal_normalized_data: Dictionary of normalized dataframes by category

    Returns:
        Dictionary of dataframes with outliers removed

    Notes:
        Configuration parameters:
        - outlier_detection: Boolean to enable/disable outlier detection
        - outliers_method: Method with which to remove outliers. Possible values are:
          - 'iforest': Uses sklearn's IsolationForest
          - 'ee': Uses sklearn's EllipticEnvelope
          - 'lof': Uses sklearn's LocalOutlierFactor
        - outlier_threshold: float, default = 0.05
          The percentage outliers to be removed from the dataset.
    """
    processed_data = {}

    # Check if outlier detection is enabled
    outlier_detection = getattr(
        context.resources.config, "outlier_detection", Defaults.OUTLIER_DETECTION
    )
    if not outlier_detection:
        context.log.info("Outlier detection disabled, returning normalized data")
        return internal_normalized_data

    # Get all configuration parameters
    outlier_threshold = getattr(
        context.resources.config, "outlier_threshold", Defaults.OUTLIER_THRESHOLD
    )
    outliers_method = getattr(context.resources.config, "outliers_method", Defaults.OUTLIER_METHOD)

    for category, df in internal_normalized_data.items():
        context.log.info(f"Removing outliers for category: {category}")

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()
        original_rows = len(pandas_df)

        # Initialize experiment
        exp = ClusteringExperiment()

        # Use configurable parameters
        setup_params = {
            "data": pandas_df,
            "remove_outliers": True,
            "outliers_method": outliers_method,
            "outliers_threshold": outlier_threshold,
            "verbose": False,
            "session_id": Defaults.SESSION_ID,
        }

        # Set up experiment
        exp.setup(**setup_params)

        # Get the outlier-cleaned data
        cleaned_df = exp.X_train_transformed
        processed_data[category] = pl.from_pandas(cleaned_df)

        # Report how many outliers were removed
        removed = original_rows - len(cleaned_df)
        context.log.info(f"Removed {removed} outliers from {category}")

    return processed_data


@dg.asset(
    name="internal_dimensionality_reduced_features",
    description="Reduces feature dimensions using PCA",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_outlier_removed_features"],
    required_resource_keys={"config"},
)
def internal_dimensionality_reduced_features(
    context: dg.AssetExecutionContext,
    internal_outlier_removed_features: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Reduce feature dimensions using PCA.

    Applies Principal Component Analysis (PCA) to reduce the dimensionality
    of features while retaining a specified amount of variance. PCA can be
    disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        internal_outlier_removed_features: Dictionary of DataFrames with outliers removed

    Returns:
        Dictionary of DataFrames with reduced dimensionality

    Notes:
        Configuration parameters:
        - pca_active: Boolean to enable/disable PCA
        - pca_components: Float (0-1) for variance retention or int for components
        - pca_method: Method to use ('linear', 'kernel', etc.)
    """
    # Check if PCA is enabled
    pca_active = getattr(context.resources.config, "pca_active", Defaults.PCA_ACTIVE)
    if not pca_active:
        context.log.info("PCA disabled, returning outlier-removed features")
        return internal_outlier_removed_features

    # Get all configuration parameters
    pca_components = getattr(context.resources.config, "pca_components", Defaults.PCA_COMPONENTS)
    pca_method = getattr(context.resources.config, "pca_method", Defaults.PCA_METHOD)

    processed_data = {}

    for category, df in internal_outlier_removed_features.items():
        context.log.info(f"Applying PCA for category: {category}")

        # Track original feature count
        original_features = df.width

        # Convert Polars DataFrame to Pandas
        pandas_df = df.to_pandas()

        # Initialize experiment
        exp = ClusteringExperiment()

        # Use configurable parameters
        setup_params = {
            "data": pandas_df,
            "pca": True,
            "pca_method": pca_method,
            "pca_components": pca_components,
            "verbose": False,
            "session_id": Defaults.SESSION_ID,
        }

        # Set up experiment
        exp.setup(**setup_params)

        # Get the PCA-transformed data
        pca_df = exp.X_train_transformed
        processed_data[category] = pl.from_pandas(pca_df)

        # Report feature reduction
        new_features = processed_data[category].width
        context.log.info(f"PCA reduced features for {category} from {original_features} to {new_features}")

    return processed_data


@dg.asset(
    name="internal_feature_metadata",
    description="Feature engineering metadata asset",
    group_name="feature_engineering",
    compute_kind="internal_feature_engineering",
    deps=["internal_dimensionality_reduced_features"],
    required_resource_keys={"config"},
)
def internal_feature_metadata(
    context: dg.AssetExecutionContext,
    internal_dimensionality_reduced_features: dict[str, pl.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Generate comprehensive metadata on engineered features.

    Creates detailed metadata for each category of processed data,
    including schema information, data shape, null counts, and optionally
    statistical summaries, sample data, correlation matrices, and preprocessing
    configuration details.

    Args:
        context: Asset execution context with access to resources and logging
        internal_dimensionality_reduced_features: Dictionary of fully processed dataframes by category

    Returns:
        Dictionary of metadata organized by category, containing information
        about the processed data to facilitate analysis and debugging

    Notes:
        The level of detail in the metadata is controlled by the
        feature_metadata_detail configuration parameter:
        - "basic": Only includes schema, shape and null counts
        - "full": Includes statistics, samples, correlations and config details
    """
    metadata = {}

    # Use metadata detail level from config or default to full
    detail_level = getattr(
        context.resources.config, "feature_metadata_detail", Defaults.METADATA_DETAIL
    )
    context.log.info(f"Generating feature metadata with detail level: {detail_level}")

    for category, df in internal_dimensionality_reduced_features.items():
        context.log.info(f"Creating metadata for category: {category}")

        # Basic metadata for all detail levels
        base_metadata = {
            "schema": df.schema,
            "shape": df.shape,
            "null_counts": df.null_count().to_dict(),
        }

        # Add detailed stats if requested
        if detail_level == "full":
            try:
                context.log.info(f"Adding detailed statistics for {category}")

                # Add descriptive statistics
                base_metadata["stats"] = df.describe().to_dicts()

                # Add sample data
                base_metadata["sample"] = df.head(5).to_dicts()

                # Add correlation matrix for numeric columns
                numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
                if numeric_cols:
                    context.log.info(
                        f"Calculating correlation matrix for {len(numeric_cols)} numeric columns"
                    )
                    corr_matrix = df.select(numeric_cols).to_pandas().corr().to_dict()
                    base_metadata["correlations"] = corr_matrix

                # Add preprocessing configuration to metadata
                # Use consistent defaults with the actual implementation
                pca_components = getattr(
                    context.resources.config, "pca_components", Defaults.PCA_COMPONENTS
                )

                base_metadata["config"] = {
                    "algorithm": getattr(context.resources.config, "algorithm", Defaults.ALGORITHM),
                    "normalization": {
                        "enabled": getattr(
                            context.resources.config, "normalize", Defaults.NORMALIZE
                        ),
                        "method": getattr(
                            context.resources.config, "norm_method", Defaults.NORM_METHOD
                        ),
                    },
                    "pca": {
                        "enabled": getattr(
                            context.resources.config, "pca_active", Defaults.PCA_ACTIVE
                        ),
                        "variance": pca_components,
                    },
                    "outlier_detection": {
                        "enabled": getattr(
                            context.resources.config,
                            "outlier_detection",
                            Defaults.OUTLIER_DETECTION,
                        ),
                        "threshold": getattr(
                            context.resources.config,
                            "outlier_threshold",
                            Defaults.OUTLIER_THRESHOLD,
                        ),
                    },
                }

            except Exception as e:
                context.log.warning(f"Error calculating detailed stats for {category}: {str(e)}")
                # Log the traceback for easier debugging
                context.log.exception("Detailed exception information:")

        metadata[category] = base_metadata
        context.log.info(f"Completed metadata generation for {category}")

    return metadata
