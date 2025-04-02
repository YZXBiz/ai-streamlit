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
    internal_clustering_output,
    internal_clusters,
    internal_need_state_data,
    internal_sales_data,
    merged_clusters,
    merged_clusters_output,
    merged_internal_data,
    normalized_internal_data,
    preprocessed_external_data,
    preprocessed_internal_sales,
    preprocessed_internal_sales_percent,
)
from clustering.dagster.resources import (
    alerts_service,
    clustering_config,
    clustering_io_manager,
    data_writer,
    logger_service,
    need_state_data_reader,
    sales_data_reader,
)
from clustering.dagster.schedules import (
    daily_internal_clustering_schedule,
    monthly_full_pipeline_schedule,
    weekly_external_clustering_schedule,
)


def load_resource_config(env: str = "dev") -> dict:
    """Load resource configuration from YAML file.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary with resource configuration
    """
    # Get the resource config file path
    resource_config_path = os.path.join(os.path.dirname(__file__), "resources", "resource_configs.yml")

    try:
        # Load the YAML file
        with open(resource_config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Get the environment-specific configuration
        env_config = config_data.get(env)
        if not env_config:
            print(f"Warning: Environment '{env}' not found in config. Using 'dev' instead.")
            env_config = config_data.get("dev", {})

        return env_config
    except Exception as e:
        print(f"Error loading resource config: {e}")
        # Return empty dict as fallback
        return {}


# Define resource config by environment
def get_resources_by_env(env: str = "dev") -> dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    # Load environment-specific configurations
    env_config = load_resource_config(env)

    # Infrastructure resources
    base_resources = {
        # IO Manager
        "io_manager": clustering_io_manager.configured(
            {"base_dir": env_config.get("io_manager", {}).get("base_dir", f"outputs/dagster_storage/{env}")}
        ),
        # Config
        "config": clustering_config.configured(
            {"env": env, "config_path": env_config.get("config", {}).get("path", "configs/internal_clustering.yml")}
        ),
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
    }

    # Client resources
    client_resources = {}
    if env_config.get("use_snowflake", False):
        from clustering.dagster.resources.clients.snowflake_client import snowflake_client

        client_resources["snowflake"] = snowflake_client

    if env_config.get("use_azure", False):
        from clustering.dagster.resources.clients.azure_storage_client import azure_blob_client

        client_resources["azure_blob"] = azure_blob_client

    # Reader resources
    reader_resources = {
        # Internal sales reader
        "input_sales_reader": sales_data_reader.configured(
            env_config.get("readers", {}).get(
                "internal_sales", {"source_type": "parquet", "path": "data/raw/internal_sales.parquet"}
            )
        ),
        # Internal need state reader
        "input_need_state_reader": need_state_data_reader.configured(
            env_config.get("readers", {}).get(
                "internal_need_state", {"source_type": "csv", "path": "data/raw/need_state.csv"}
            )
        ),
        # External sales reader
        "input_external_sales_reader": sales_data_reader.configured(
            env_config.get("readers", {}).get(
                "external_sales", {"source_type": "parquet", "path": "data/raw/external_sales.parquet"}
            )
        ),
    }

    # Writer resources
    writer_resources = {
        # Internal sales output
        "output_sales_writer": data_writer.configured(
            env_config.get("writers", {}).get(
                "internal_sales_output",
                {"destination_type": "parquet", "path": "data/processed/internal_sales_processed.parquet"},
            )
        ),
        # Internal sales percent output
        "output_sales_percent_writer": data_writer.configured(
            env_config.get("writers", {}).get(
                "internal_sales_percent_output",
                {"destination_type": "parquet", "path": "data/processed/internal_sales_percent.parquet"},
            )
        ),
        # Internal clusters output
        "output_clusters_writer": data_writer.configured(
            env_config.get("writers", {}).get(
                "internal_clusters_output",
                {"destination_type": "parquet", "path": "data/processed/internal_clusters.parquet"},
            )
        ),
    }

    # Combine all resources
    all_resources = {
        **base_resources,
        **client_resources,
        **reader_resources,
        **writer_resources,
    }

    return all_resources


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
            normalized_internal_data,
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
            normalized_internal_data,
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
            normalized_internal_data,
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
        schedules=[
            daily_internal_clustering_schedule,
            weekly_external_clustering_schedule,
            monthly_full_pipeline_schedule,
        ],
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
