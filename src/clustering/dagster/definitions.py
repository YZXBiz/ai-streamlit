"""Dagster definitions module for the clustering pipeline."""

import os
from types import SimpleNamespace
from typing import Any

import dagster as dg
import yaml

# -----------------------------------------------------------------------------
# Asset imports
# -----------------------------------------------------------------------------
# Internal preprocessing assets
# External preprocessing assets
# Internal ML assets - feature engineering
# Internal ML assets - model training and analysis
# External ML assets - feature engineering
# External ML assets - model training and analysis
# Merging assets
from clustering.dagster.assets import (
    cluster_reassignment,
    external_assign_clusters,
    external_dimensionality_reduced_features,
    external_fe_raw_data,
    external_features_data,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_optimal_cluster_counts,
    external_outlier_removed_features,
    external_save_cluster_assignments,
    external_save_clustering_models,
    external_train_clustering_models,
    internal_assign_clusters,
    internal_dimensionality_reduced_features,
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_normalized_sales_data,
    internal_optimal_cluster_counts,
    internal_outlier_removed_features,
    internal_output_sales_table,
    internal_product_category_mapping,
    internal_raw_sales_data,
    internal_sales_by_category,
    internal_sales_with_categories,
    internal_save_cluster_assignments,
    internal_save_clustering_models,
    internal_train_clustering_models,
    merged_cluster_assignments,
    merged_clusters,
    optimized_merged_clusters,
    preprocessed_external_data,
    save_merged_cluster_assignments,
)

# Resources
from clustering.dagster.resources import data_io, logger_service
from clustering.infra import Environment

# Import Hydra-style config loader from the new location in infra
from clustering.infra.hydra_config import load_config as hydra_load_config

# -----------------------------------------------------------------------------
# Asset selection lists
# -----------------------------------------------------------------------------

# Group assets by categories to avoid redundancy in job definitions
internal_preprocessing_assets = [
    internal_raw_sales_data,
    internal_product_category_mapping,
    internal_sales_with_categories,
    internal_normalized_sales_data,
    internal_sales_by_category,
    internal_output_sales_table,
]

external_preprocessing_assets = [
    external_features_data,
    preprocessed_external_data,
]

internal_feature_engineering_assets = [
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
    internal_dimensionality_reduced_features,
]

internal_model_training_assets = [
    internal_optimal_cluster_counts,
    internal_train_clustering_models,
    internal_save_clustering_models,
]

internal_cluster_assignment_assets = [
    internal_assign_clusters,
    internal_save_cluster_assignments,
]

external_feature_engineering_assets = [
    external_fe_raw_data,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_outlier_removed_features,
    external_dimensionality_reduced_features,
]

external_model_training_assets = [
    external_optimal_cluster_counts,
    external_train_clustering_models,
    external_save_clustering_models,
]

external_cluster_assignment_assets = [
    external_assign_clusters,
    external_save_cluster_assignments,
]

merging_assets_list = [
    merged_clusters,
    merged_cluster_assignments,
    optimized_merged_clusters,
    cluster_reassignment,
    save_merged_cluster_assignments,
]

# -----------------------------------------------------------------------------
# Job definitions
# -----------------------------------------------------------------------------

# 1. Internal preprocessing job
internal_preprocessing_job = dg.define_asset_job(
    name="internal_preprocessing_job",
    selection=internal_preprocessing_assets,
    tags={"kind": "internal_preprocessing"},
)

# 2. External preprocessing job
external_preprocessing_job = dg.define_asset_job(
    name="external_preprocessing_job",
    selection=external_preprocessing_assets,
    tags={"kind": "external_preprocessing"},
)

# 3. Internal ML job
internal_ml_job = dg.define_asset_job(
    name="internal_ml_job",
    selection=[
        # Feature engineering assets
        *internal_feature_engineering_assets,
        # Model training assets
        *internal_model_training_assets,
        # Cluster assignment
        *internal_cluster_assignment_assets,
    ],
    tags={"kind": "internal_ml"},
)

# 4. External ML job
external_ml_job = dg.define_asset_job(
    name="external_ml_job",
    selection=[
        # Feature engineering assets
        *external_feature_engineering_assets,
        # Model training assets
        *external_model_training_assets,
        # Cluster assignment
        *external_cluster_assignment_assets,
    ],
    tags={"kind": "external_ml"},
)

# 5. Merging job
merging_job = dg.define_asset_job(
    name="merging_job",
    selection=merging_assets_list,
    tags={"kind": "merging"},
)

# 6. Full pipeline job (combining all jobs)
full_pipeline_job = dg.define_asset_job(
    name="full_pipeline_job",
    selection=[
        # Internal preprocessing
        *internal_preprocessing_assets,
        # Internal feature engineering
        *internal_feature_engineering_assets,
        # Internal model training
        *internal_model_training_assets,
        # Internal cluster assignment
        *internal_cluster_assignment_assets,
        # External preprocessing
        *external_preprocessing_assets,
        # External feature engineering
        *external_feature_engineering_assets,
        # External model training
        *external_model_training_assets,
        # External cluster assignment
        *external_cluster_assignment_assets,
        # Merging
        *merging_assets_list,
    ],
    tags={"kind": "complete_pipeline"},
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 1  # Sequential execution
                }
            }
        }
    },
)

# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------


def load_config(env: str = "dev") -> dict[str, Any]:
    """Load configuration from YAML file for the specified environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary containing configuration data with resolved environment variables
    """
    config_path = os.path.join(os.path.dirname(__file__), "resources", "configs", f"{env}.yml")

    try:
        # Use the Hydra-style config loader to resolve environment variables
        config_data = hydra_load_config(config_path)

        if config_data is None:
            print(f"WARNING: Config file {config_path} parsed as None, using empty config")
            config_data = {}
    except FileNotFoundError as e:
        print(f"ERROR: Config file {config_path} not found: {str(e)}")
        config_data = _get_default_config(env)
    except (yaml.YAMLError, yaml.parser.ParserError) as e:
        print(f"ERROR: Invalid YAML in config file {config_path}: {str(e)}")
        config_data = _get_default_config(env)
    except Exception as e:
        print(f"ERROR loading config from {config_path}: {str(e)}")
        config_data = _get_default_config(env)

    return config_data


def _get_default_config(env: str) -> dict[str, Any]:
    """Get default configuration when config loading fails.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Default configuration dictionary
    """
    return {
        "job_params": {},
        "logger": {"level": "INFO", "sink": f"logs/dagster_{env}.log"},
        "readers": {},
        "writers": {},
    }


# -----------------------------------------------------------------------------
# Resource definitions
# -----------------------------------------------------------------------------


def get_resources_by_env(
    env: str | Environment = Environment.DEV,
) -> dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    # Convert Environment enum to string if needed
    env_str = env.value if isinstance(env, Environment) else env

    # Load configuration
    config_data = load_config(env_str)

    # Extract configuration sections
    job_params = config_data.get("job_params", {})
    logger_config = config_data.get("logger", {})
    readers_config = config_data.get("readers", {})
    writers_config = config_data.get("writers", {})

    # Create config object
    params = SimpleNamespace(**job_params)
    # Set the env as Environment enum if possible
    try:
        params.env = Environment(env_str)
    except ValueError:
        # Fallback to string if not a valid enum value
        params.env = env_str

    # Create the params resource
    params_resource = dg.resource(lambda: params)()

    # Define all resources
    resources = {
        # Core resources
        "io_manager": dg.FilesystemIOManager(
            base_dir=os.environ.get("DAGSTER_STORAGE_DIR", "storage")
        ),
        # Parameter resources (both names point to same resource)
        "job_params": params_resource,
        "config": params_resource,
        # Logger
        "logger": logger_service.configured(
            {
                "sink": logger_config.get("sink", f"logs/dagster_{env_str}.log"),
                "level": logger_config.get("level", "INFO"),
            }
        ),
        # Data readers
        "internal_ns_sales": data_io.data_reader.configured(readers_config.get("ns_sales", {})),
        "internal_ns_map": data_io.data_reader.configured(readers_config.get("ns_map", {})),
        "sales_by_category_reader": data_io.data_reader.configured(
            readers_config.get("sales_by_category", {})
        ),
        "external_data_reader": data_io.data_reader.configured(
            readers_config.get("external_data_source", {})
        ),
        "input_external_placerai_reader": data_io.data_reader.configured(
            readers_config.get("external_placerai", {})
        ),
        # Data writers
        "sales_by_category_writer": data_io.data_writer.configured(
            writers_config.get("sales_by_category", {})
        ),
        "output_clusters_writer": data_io.data_writer.configured(
            writers_config.get("internal_clusters_output", {})
        ),
        "output_external_data_writer": data_io.data_writer.configured(
            writers_config.get("external_data_output", {})
        ),
        "internal_model_output": data_io.data_writer.configured(
            writers_config.get("model_output", {})
        ),
        "internal_cluster_assignments": data_io.data_writer.configured(
            writers_config.get("cluster_assignments", {})
        ),
        "external_model_output": data_io.data_writer.configured(
            writers_config.get("external_model_output", {})
        ),
        "external_cluster_assignments": data_io.data_writer.configured(
            writers_config.get("external_cluster_assignments", {})
        ),
        "merged_cluster_assignments": data_io.data_writer.configured(
            writers_config.get("merged_cluster_assignments", {})
        ),
    }

    return resources


# -----------------------------------------------------------------------------
# Definitions creation
# -----------------------------------------------------------------------------


def create_definitions(env: str | Environment = Environment.DEV) -> dg.Definitions:
    """Create Dagster definitions for the specified environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dagster definitions object containing assets, resources, and jobs
    """
    # Get resources for the environment
    resources = get_resources_by_env(env)

    # Collect all assets in a single list
    assets = [
        *internal_preprocessing_assets,
        *external_preprocessing_assets,
        *internal_feature_engineering_assets,
        *external_feature_engineering_assets,
        *internal_model_training_assets,
        *external_model_training_assets,
        *internal_cluster_assignment_assets,
        *external_cluster_assignment_assets,
        *merging_assets_list,
    ]

    # Create and return definitions with all assets, resources, and jobs
    return dg.Definitions(
        assets=assets,
        resources=resources,
        jobs=[
            internal_preprocessing_job,
            external_preprocessing_job,
            internal_ml_job,
            external_ml_job,
            merging_job,
            full_pipeline_job,
        ],
    )


# -----------------------------------------------------------------------------
# Default definitions
# -----------------------------------------------------------------------------

# Create default definitions for dev environment
defs = create_definitions(env=Environment.DEV)


def get_definitions():
    """Get the definitions object.

    Returns:
        Dagster definitions object for the default (dev) environment
    """
    return defs


# Export symbols
__all__ = ["create_definitions", "get_definitions", "defs"]
