"""Feature engineering assets for external data using PyCaret with Dagster config system."""


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
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["preprocessed_external_data"],
    required_resource_keys={"external_data_reader"},
)
def external_fe_raw_data(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load raw external data using the configured reader resource.

    This asset depends on preprocessed_external_data to ensure the preprocessing
    pipeline completes before feature engineering starts.

    Args:
        context: Asset execution context with access to resources and logging

    Returns:
        DataFrame with external data
    """
    context.log.info("Loading external data by category")
    return context.resources.external_data_reader.read()


@dg.asset(
    name="external_filtered_features",
    description="Filters out ignored features from raw external data",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_fe_raw_data"],
    required_resource_keys={"config"},
)
def external_filtered_features(
    context: dg.AssetExecutionContext,
    external_fe_raw_data: pl.DataFrame,
) -> pl.DataFrame:
    """Filter out features that should be ignored using PyCaret's ignore_features.

    Uses the configuration to determine which features should be excluded from
    the analysis. Features listed in the ignore_features config will be removed
    from all category dataframes where they exist.

    Args:
        context: Asset execution context with access to resources and logging
        external_fe_raw_data: DataFrame with raw data from external sources

    Returns:
        DataFrame with specified features filtered out
    """
    # Get features to ignore from context resources
    ignore_features = getattr(context.resources.config, "ignore_features", [])

    if not ignore_features:
        context.log.info("No features to ignore, returning raw data")
        return external_fe_raw_data

    context.log.info(f"Features to ignore: {ignore_features}")

    # Storing original features for metadata
    original_features = external_fe_raw_data.columns

    # Get features that actually exist in this dataframe
    features_to_ignore = [col for col in ignore_features if col in external_fe_raw_data.columns]

    if not features_to_ignore:
        context.log.info("No features to ignore")
        return external_fe_raw_data

    context.log.info(f"Ignoring features via PyCaret setup: {features_to_ignore}")

    # Convert Polars DataFrame to Pandas
    pandas_df = external_fe_raw_data.to_pandas()

    # Initialize PyCaret experiment with ignore_features
    exp = ClusteringExperiment()
    exp.setup(
        data=pandas_df,
        ignore_features=features_to_ignore,
        session_id=Defaults.SESSION_ID,
        verbose=False,
    )

    # Get the transformed data with ignored features removed
    filtered_data = pl.from_pandas(exp.X_train_transformed)

    context.log.info(f"Removed features: {features_to_ignore}")

    # Store the ignored features in the context metadata for potential later use
    context.add_output_metadata(
        {"ignored_features": ignore_features, "original_features": original_features}
    )

    return filtered_data


@dg.asset(
    name="external_imputed_features",
    description="Imputes missing values in external features using PyCaret",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_filtered_features"],
    required_resource_keys={"config"},
)
def external_imputed_features(
    context: dg.AssetExecutionContext,
    external_filtered_features: pl.DataFrame,
) -> pl.DataFrame:
    """Impute missing values in external features using PyCaret's imputation methods.

    Handles missing values in the dataframe using the configured
    imputation strategy. Imputation is configurable for both numeric and
    categorical features.

    Args:
        context: Asset execution context with access to resources and logging
        external_filtered_features: DataFrame with filtered features

    Returns:
        DataFrame with missing values imputed

    Notes:
        Configuration parameters:
        - imputation_type: Options are 'simple' or 'iterative'
        - numeric_imputation: Method for numeric features ('mean', 'median', etc.)
        - categorical_imputation: Method for categorical features ('mode', etc.)
    """
    # Get all configuration parameters with defaults
    imputation_type = getattr(context.resources.config, "imputation_type", Defaults.IMPUTATION_TYPE)
    numeric_imputation = getattr(
        context.resources.config, "numeric_imputation", Defaults.NUMERIC_IMPUTATION
    )
    categorical_imputation = getattr(
        context.resources.config, "categorical_imputation", Defaults.CATEGORICAL_IMPUTATION
    )

    context.log.info("Imputing missing values")

    # Store original data
    original_data = external_filtered_features

    # Convert Polars DataFrame to Pandas
    pandas_df = external_filtered_features.to_pandas()

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
    processed_data = pl.from_pandas(exp.X_train_transformed)

    context.log.info("Missing value imputation completed")

    # Store original data in context for potential restoration of ignored features
    # Properly wrap the tuple with MetadataValue.json
    context.add_output_metadata(
        {"original_data_shape": dg.MetadataValue.json(list(original_data.shape))}
    )

    return processed_data


@dg.asset(
    name="external_normalized_data",
    description="Applies feature scaling/normalization to external data using PyCaret",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_imputed_features"],
    required_resource_keys={"config"},
)
def external_normalized_data(
    context: dg.AssetExecutionContext,
    external_imputed_features: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize features using PyCaret's normalization methods.

    Applies feature scaling/normalization to numeric features based on the
    configured method. Normalization can be disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        external_imputed_features: DataFrame with imputed features

    Returns:
        DataFrame with normalized features

    Notes:
        Configuration parameters:
        - normalize: Boolean to enable/disable normalization
        - norm_method: Method to use for normalization ('robust', 'zscore', etc.)
    """
    # Check if normalization is enabled
    normalize = getattr(context.resources.config, "normalize", Defaults.NORMALIZE)
    if not normalize:
        context.log.info("Normalization disabled, returning imputed features")
        return external_imputed_features

    # Get normalization method
    norm_method = getattr(context.resources.config, "norm_method", Defaults.NORM_METHOD)

    context.log.info("Normalizing features")

    # Convert Polars DataFrame to Pandas
    pandas_df = external_imputed_features.to_pandas()

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

    processed_data = pl.from_pandas(exp.X_train_transformed)

    context.log.info("Normalization completed")

    return processed_data


@dg.asset(
    name="external_outlier_removed_features",
    description="Detects and removes outliers from external data",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_normalized_data"],
    required_resource_keys={"config"},
)
def external_outlier_removed_features(
    context: dg.AssetExecutionContext,
    external_normalized_data: pl.DataFrame,
) -> pl.DataFrame:
    """Detect and remove outliers using isolation forest or other methods.

    Applies outlier detection and removal using PyCaret. It processes
    the dataframe by detecting and removing outliers according
    to the configured method and threshold.

    Args:
        context: Asset execution context with access to resources and logging
        external_normalized_data: DataFrame with normalized features

    Returns:
        DataFrame with outliers removed

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
    # Check if outlier detection is enabled
    outlier_detection = getattr(
        context.resources.config, "outlier_detection", Defaults.OUTLIER_DETECTION
    )
    if not outlier_detection:
        context.log.info("Outlier detection disabled, returning normalized data")
        return external_normalized_data

    # Get all configuration parameters
    outlier_threshold = getattr(
        context.resources.config, "outlier_threshold", Defaults.OUTLIER_THRESHOLD
    )
    outliers_method = getattr(context.resources.config, "outliers_method", Defaults.OUTLIER_METHOD)

    context.log.info("Removing outliers")

    # Convert Polars DataFrame to Pandas
    pandas_df = external_normalized_data.to_pandas()
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
    processed_data = pl.from_pandas(cleaned_df)

    # Report how many outliers were removed
    removed = original_rows - len(cleaned_df)
    context.log.info(f"Removed {removed} outliers")

    return processed_data


@dg.asset(
    name="external_dimensionality_reduced_features",
    description="Reduces external feature dimensions using PCA",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_outlier_removed_features"],
    required_resource_keys={"config"},
)
def external_dimensionality_reduced_features(
    context: dg.AssetExecutionContext,
    external_outlier_removed_features: pl.DataFrame,
) -> pl.DataFrame:
    """Reduce feature dimensions using PCA.

    Applies Principal Component Analysis (PCA) to reduce the dimensionality
    of features while retaining a specified amount of variance. PCA can be
    disabled via configuration.

    Args:
        context: Asset execution context with access to resources and logging
        external_outlier_removed_features: DataFrame with outliers removed

    Returns:
        DataFrame with reduced feature dimensions

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
        return external_outlier_removed_features

    # Get all configuration parameters
    pca_components = getattr(context.resources.config, "pca_components", Defaults.PCA_COMPONENTS)
    pca_method = getattr(context.resources.config, "pca_method", Defaults.PCA_METHOD)

    context.log.info("Applying PCA")

    # Track original feature count
    original_features = external_outlier_removed_features.width

    # Convert Polars DataFrame to Pandas
    pandas_df = external_outlier_removed_features.to_pandas()

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
    processed_data = pl.from_pandas(pca_df)

    # Report feature reduction
    new_features = processed_data.width
    context.log.info(f"PCA reduced features from {original_features} to {new_features}")

    return processed_data


@dg.asset(
    name="external_feature_metadata",
    description="External feature engineering metadata asset",
    group_name="feature_engineering",
    compute_kind="external_feature_engineering",
    deps=["external_dimensionality_reduced_features"],
    required_resource_keys={"config"},
)
def external_feature_metadata(
    context: dg.AssetExecutionContext,
    external_dimensionality_reduced_features: pl.DataFrame,
) -> pl.DataFrame:
    """Generate comprehensive metadata on engineered features.

    Creates detailed metadata for the processed data,
    including schema information, data shape, null counts, and optionally
    statistical summaries, sample data, correlation matrices, and preprocessing
    configuration details.

    Args:
        context: Asset execution context with access to resources and logging
        external_dimensionality_reduced_features: DataFrame with reduced feature dimensions

    Returns:
        DataFrame containing metadata about the processed data

    Notes:
        The level of detail in the metadata is controlled by the
        feature_metadata_detail configuration parameter:
        - "basic": Only includes schema, shape and null counts
        - "full": Includes statistics, samples, correlations and config details
    """
    # Use metadata detail level from config or default to full
    detail_level = getattr(
        context.resources.config, "feature_metadata_detail", Defaults.METADATA_DETAIL
    )
    context.log.info(f"Generating feature metadata with detail level: {detail_level}")

    df = external_dimensionality_reduced_features
    context.log.info("Creating metadata")

    # Basic metadata for all detail levels
    base_metadata = {
        "schema": str(df.schema),
        "shape": str(df.shape),
        "null_counts": str(df.null_count().to_dict()),
    }

    # Create a DataFrame from the metadata
    metadata_df = pl.DataFrame([base_metadata])

    # Add detailed stats if requested
    if detail_level == "full":
        try:
            context.log.info("Adding detailed statistics")

            # Add descriptive statistics as string columns
            stats_dict = {f"stats_{i}": str(v) for i, v in enumerate(df.describe().to_dicts())}
            metadata_df = metadata_df.with_columns(
                [pl.lit(v).alias(k) for k, v in stats_dict.items()]
            )

            # Add sample data
            sample_dict = {f"sample_{i}": str(v) for i, v in enumerate(df.head(5).to_dicts())}
            metadata_df = metadata_df.with_columns(
                [pl.lit(v).alias(k) for k, v in sample_dict.items()]
            )

            # Add correlation matrix for numeric columns
            numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
            if numeric_cols:
                context.log.info(
                    f"Calculating correlation matrix for {len(numeric_cols)} numeric columns"
                )
                corr_matrix = df.select(numeric_cols).to_pandas().corr().to_dict()
                metadata_df = metadata_df.with_column(
                    pl.lit(str(corr_matrix)).alias("correlations")
                )

            # Add preprocessing configuration to metadata
            # Use consistent defaults with the actual implementation
            pca_components = getattr(
                context.resources.config, "pca_components", Defaults.PCA_COMPONENTS
            )

            config_info = {
                "algorithm": getattr(context.resources.config, "algorithm", Defaults.ALGORITHM),
                "normalization": {
                    "enabled": getattr(context.resources.config, "normalize", Defaults.NORMALIZE),
                    "method": getattr(
                        context.resources.config, "norm_method", Defaults.NORM_METHOD
                    ),
                },
                "pca": {
                    "enabled": getattr(context.resources.config, "pca_active", Defaults.PCA_ACTIVE),
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

            metadata_df = metadata_df.with_column(pl.lit(str(config_info)).alias("config"))

        except Exception as e:
            context.log.warning(f"Error calculating detailed stats: {str(e)}")
            # Log the traceback for easier debugging
            context.log.exception("Detailed exception information:")

    context.log.info("Completed metadata generation")

    return metadata_df
