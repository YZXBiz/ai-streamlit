"""Dagster definitions module for the clustering pipeline."""

import os

import dagster as dg
import yaml

from clustering.dagster.assets import (
    external_cluster_evaluation,
    external_clustering_model,
    external_clustering_output,
    external_clusters,
    external_features_data,
    internal_category_data,
    internal_cluster_evaluation,
    internal_clustering_model,
    internal_clustering_output,
    internal_clusters,
    internal_need_state_data,
    internal_sales_data,
    merged_clusters,
    merged_clusters_output,
    merged_internal_data,
    preprocessed_external_data,
    preprocessed_internal_sales,
    preprocessed_internal_sales_percent,
)
from clustering.dagster.resources import (
    alerts_service,
    clustering_io_manager,
    data_io,
    logger_service,
    simple_config,
)


def load_resource_config(env: str = "dev") -> dict:
    """Load resource configuration from YAML file.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary with resource configuration
    """
    # Get the resource config file path
    config_file_path = os.path.join(os.path.dirname(__file__), "resources", "configs", f"{env}.yml")

    try:
        # Load the YAML file
        with open(config_file_path) as f:
            config_data = yaml.safe_load(f)

        return config_data
    except Exception as e:
        print(f"Error loading resource config for environment '{env}': {e}")
        return {}


# Define resources with a flat structure
def get_resources_by_env(env: str = "dev") -> dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    # Load environment-specific configuration
    env_config = load_resource_config(env)

    # Create all resources in a flat structure
    resources = {
        # Core resources
        "io_manager": clustering_io_manager,
        "config": simple_config.configured({"env": env}),
        # Logger
        "logger": logger_service.configured(
            {
                "sink": env_config.get("logger", {}).get("sink", f"logs/dagster_{env}.log"),
                "level": env_config.get("logger", {}).get("level", "INFO"),
            }
        ),
        # Alerts
        "alerts": alerts_service.configured(
            {
                "enabled": env_config.get("alerts", {}).get("enabled", True),
                "threshold": env_config.get("alerts", {}).get("threshold", "WARNING"),
                "slack_webhook": env_config.get("alerts", {}).get("slack_webhook", None),
            }
        ),
        # Data readers
        "input_sales_reader": data_io.data_reader.configured(
            env_config.get("readers", {}).get("internal_sales", {})
        ),
        "input_need_state_reader": data_io.data_reader.configured(
            env_config.get("readers", {}).get("internal_need_state", {})
        ),
        "input_external_sales_reader": data_io.data_reader.configured(
            env_config.get("readers", {}).get("external_sales", {})
        ),
        # Data writers
        "output_sales_writer": data_io.data_writer.configured(
            env_config.get("writers", {}).get("internal_sales_output", {})
        ),
        "output_sales_percent_writer": data_io.data_writer.configured(
            env_config.get("writers", {}).get("internal_sales_percent_output", {})
        ),
        "output_clusters_writer": data_io.data_writer.configured(
            env_config.get("writers", {}).get("internal_clusters_output", {})
        ),
        "output_external_data_writer": data_io.data_writer.configured(
            env_config.get("writers", {}).get("external_data_output", {})
        ),
        "output_merged_writer": data_io.data_writer.configured(
            env_config.get("writers", {}).get("merged_clusters_output", {})
        ),
    }

    return resources


# Define asset jobs
def define_internal_preprocessing_job() -> dg.AssetsDefinition:
    """Define the internal preprocessing job.

    Returns:
        Asset job for internal preprocessing
    """
    return dg.define_asset_job(
        name="internal_preprocessing_job",
        selection=[
            internal_sales_data,
            internal_need_state_data,
            merged_internal_data,
            internal_category_data,
            preprocessed_internal_sales,
            preprocessed_internal_sales_percent,
        ],
        tags={"kind": "internal_preprocessing"},
    )


def define_internal_clustering_job() -> dg.AssetsDefinition:
    """Define the internal clustering job.

    Returns:
        Asset job for internal clustering
    """
    return dg.define_asset_job(
        name="internal_clustering_job",
        selection=[
            internal_clustering_model,
            internal_clusters,
            internal_cluster_evaluation,
            internal_clustering_output,
        ],
        tags={"kind": "internal_clustering"},
    )


def define_external_preprocessing_job() -> dg.AssetsDefinition:
    """Define the external preprocessing job.

    Returns:
        Asset job for external preprocessing
    """
    return dg.define_asset_job(
        name="external_preprocessing_job",
        selection=[
            external_features_data,
            preprocessed_external_data,
        ],
        tags={"kind": "external_preprocessing"},
    )


def define_external_clustering_job() -> dg.AssetsDefinition:
    """Define the external clustering job.

    Returns:
        Asset job for external clustering
    """
    return dg.define_asset_job(
        name="external_clustering_job",
        selection=[
            external_clustering_model,
            external_clusters,
            external_cluster_evaluation,
            external_clustering_output,
        ],
        tags={"kind": "external_clustering"},
    )


def define_merging_job() -> dg.AssetsDefinition:
    """Define the merging job.

    Returns:
        Asset job for merging internal and external clusters
    """
    return dg.define_asset_job(
        name="merging_job",
        selection=[
            merged_clusters,
            merged_clusters_output,
        ],
        tags={"kind": "merging"},
    )


def define_full_pipeline_job() -> dg.AssetsDefinition:
    """Define the full pipeline job that runs all assets.

    Returns:
        Asset job for the full pipeline
    """
    return dg.define_asset_job(
        name="full_pipeline_job",
        selection=[
            # Internal preprocessing
            internal_sales_data,
            internal_need_state_data,
            merged_internal_data,
            internal_category_data,
            preprocessed_internal_sales,
            preprocessed_internal_sales_percent,
            # Internal clustering
            internal_clustering_model,
            internal_clusters,
            internal_cluster_evaluation,
            internal_clustering_output,
            # External preprocessing
            external_features_data,
            preprocessed_external_data,
            # External clustering
            external_clustering_model,
            external_clusters,
            external_cluster_evaluation,
            external_clustering_output,
            # Merging
            merged_clusters,
            merged_clusters_output,
        ],
        tags={"kind": "full_pipeline"},
    )


# Define the full Dagster definitions
def create_definitions(env: str = "dev") -> dg.Definitions:
    """Create Dagster definitions.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dagster definitions
    """
    resources = get_resources_by_env(env)

    # Define asset jobs
    internal_preprocessing_job = define_internal_preprocessing_job()
    internal_clustering_job = define_internal_clustering_job()
    external_preprocessing_job = define_external_preprocessing_job()
    external_clustering_job = define_external_clustering_job()
    merging_job = define_merging_job()
    full_pipeline_job = define_full_pipeline_job()

    # Create and return definitions
    return dg.Definitions(
        assets=[
            # Preprocessing assets - Internal
            internal_sales_data,
            internal_need_state_data,
            merged_internal_data,
            internal_category_data,
            preprocessed_internal_sales,
            preprocessed_internal_sales_percent,
            # Preprocessing assets - External
            external_features_data,
            preprocessed_external_data,
            # Clustering assets - Internal
            internal_clustering_model,
            internal_clusters,
            internal_cluster_evaluation,
            internal_clustering_output,
            # Clustering assets - External
            external_clustering_model,
            external_clusters,
            external_cluster_evaluation,
            external_clustering_output,
            # Merging assets
            merged_clusters,
            merged_clusters_output,
        ],
        resources=resources,
        jobs=[
            internal_preprocessing_job,
            internal_clustering_job,
            external_preprocessing_job,
            external_clustering_job,
            merging_job,
            full_pipeline_job,
        ],
    )


# Create default definitions with dev environment
defs = create_definitions(env="dev")
