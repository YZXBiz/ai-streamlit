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
    data_io,
    logger_service,
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
    config_file_path = os.path.join(os.path.dirname(__file__), "resources", "configs", f"{env}.yml")

    try:
        # Load the YAML file
        with open(config_file_path, "r") as f:
            config_data = yaml.safe_load(f)

        return config_data
    except Exception as e:
        print(f"Error loading resource config for environment '{env}': {e}")
        # Fall back to base config
        base_config_path = os.path.join(os.path.dirname(__file__), "resources", "configs", "base.yml")
        try:
            with open(base_config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as base_error:
            print(f"Error loading base config: {base_error}")
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
            {
                "base_dir": os.path.join("outputs", env),
            }
        ),
        # Config
        "config": clustering_config.configured(
            {
                "env": env,
                # Use the job config files from the root configs directory, not the old unused structure
                "config_path": env_config.get("config", {}).get(
                    "path",
                    os.path.join(
                        os.path.dirname(__file__), "resources", "configs", "job_configs", "internal_clustering.yml"
                    ),
                ),
            }
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

    # Configure data reader and writer resources based on environment config
    io_resources = {
        # Internal sales reader
        "input_sales_reader": data_io.data_reader.configured(
            {
                "kind": "ParquetReader" if env == "dev" else "BlobReader" if env == "staging" else "SnowflakeReader",
                "config": {
                    "path": env_config.get("readers", {})
                    .get("internal_sales", {})
                    .get("path", "data/raw/internal_sales.parquet")
                }
                if env == "dev"
                else (
                    {
                        "container": env_config.get("readers", {})
                        .get("internal_sales", {})
                        .get("container", "data-container"),
                        "blob_path": env_config.get("readers", {})
                        .get("internal_sales", {})
                        .get("blob_path", "internal/sales.parquet"),
                    }
                    if env == "staging"
                    else {
                        "query": env_config.get("readers", {})
                        .get("internal_sales", {})
                        .get("query", "SELECT * FROM PROD_DB.RAW.INTERNAL_SALES")
                    }
                ),
            }
        ),
        # Internal need state reader
        "input_need_state_reader": data_io.data_reader.configured(
            {
                "kind": "CSVReader" if env == "dev" else "BlobReader" if env == "staging" else "SnowflakeReader",
                "config": {
                    "path": env_config.get("readers", {})
                    .get("internal_need_state", {})
                    .get("path", "data/raw/need_state.csv")
                }
                if env == "dev"
                else (
                    {
                        "container": env_config.get("readers", {})
                        .get("internal_need_state", {})
                        .get("container", "data-container"),
                        "blob_path": env_config.get("readers", {})
                        .get("internal_need_state", {})
                        .get("blob_path", "internal/need_state.csv"),
                    }
                    if env == "staging"
                    else {
                        "query": env_config.get("readers", {})
                        .get("internal_need_state", {})
                        .get("query", "SELECT * FROM PROD_DB.RAW.INTERNAL_NEED_STATE")
                    }
                ),
            }
        ),
        # External sales reader
        "input_external_sales_reader": data_io.data_reader.configured(
            {
                "kind": "ParquetReader" if env == "dev" else "SnowflakeReader",
                "config": {
                    "path": env_config.get("readers", {})
                    .get("external_sales", {})
                    .get("path", "data/raw/external_sales.parquet")
                }
                if env == "dev"
                else {
                    "query": env_config.get("readers", {})
                    .get("external_sales", {})
                    .get("query", "SELECT * FROM STAGING_CLUSTERING_DB.RAW.EXTERNAL_SALES")
                },
            }
        ),
        # Output writers using the data_writer resource
        "output_sales_writer": data_io.data_writer.configured(
            {
                "kind": "ParquetWriter" if env == "dev" else "BlobWriter" if env == "staging" else "SnowflakeWriter",
                "config": {
                    "path": env_config.get("writers", {})
                    .get("internal_sales_output", {})
                    .get("path", "data/processed/internal_sales_processed.parquet")
                }
                if env == "dev"
                else (
                    {
                        "container": env_config.get("writers", {})
                        .get("internal_sales_output", {})
                        .get("container", "processed-container"),
                        "blob_path": env_config.get("writers", {})
                        .get("internal_sales_output", {})
                        .get("blob_path", "internal/sales_processed.parquet"),
                    }
                    if env == "staging"
                    else {
                        "table_name": env_config.get("writers", {})
                        .get("internal_sales_output", {})
                        .get("table_name", "PROD_DB.PROCESSED.INTERNAL_SALES")
                    }
                ),
            }
        ),
        # Internal sales percent output writer
        "output_sales_percent_writer": data_io.data_writer.configured(
            {
                "kind": "ParquetWriter" if env == "dev" else "BlobWriter" if env == "staging" else "SnowflakeWriter",
                "config": {
                    "path": env_config.get("writers", {})
                    .get("internal_sales_percent_output", {})
                    .get("path", "data/processed/internal_sales_percent.parquet")
                }
                if env == "dev"
                else (
                    {
                        "container": env_config.get("writers", {})
                        .get("internal_sales_percent_output", {})
                        .get("container", "processed-container"),
                        "blob_path": env_config.get("writers", {})
                        .get("internal_sales_percent_output", {})
                        .get("blob_path", "internal/sales_percent.parquet"),
                    }
                    if env == "staging"
                    else {
                        "table_name": env_config.get("writers", {})
                        .get("internal_sales_percent_output", {})
                        .get("table_name", "PROD_DB.PROCESSED.INTERNAL_SALES_PERCENT")
                    }
                ),
            }
        ),
        # Internal clusters output writer
        "output_clusters_writer": data_io.data_writer.configured(
            {
                "kind": "ParquetWriter" if env == "dev" else "SnowflakeWriter",
                "config": {
                    "path": env_config.get("writers", {})
                    .get("internal_clusters_output", {})
                    .get("path", "data/processed/internal_clusters.parquet")
                }
                if env == "dev"
                else {
                    "table_name": env_config.get("writers", {})
                    .get("internal_clusters_output", {})
                    .get("table_name", "STAGING_CLUSTERING_DB.PROCESSED.INTERNAL_CLUSTERS")
                },
            }
        ),
        # External data output writer
        "output_external_data_writer": data_io.data_writer.configured(
            {
                "kind": "ParquetWriter" if env == "dev" else "SnowflakeWriter",
                "config": {
                    "path": env_config.get("writers", {})
                    .get("external_data_output", {})
                    .get("path", "data/processed/external_data_processed.parquet")
                }
                if env == "dev"
                else {
                    "table_name": env_config.get("writers", {})
                    .get("external_data_output", {})
                    .get("table_name", "STAGING_CLUSTERING_DB.PROCESSED.EXTERNAL_DATA")
                },
            }
        ),
        # External clusters output writer
        "output_merged_writer": data_io.data_writer.configured(
            {
                "kind": "ParquetWriter" if env == "dev" else "SnowflakeWriter",
                "config": {
                    "path": env_config.get("writers", {})
                    .get("merged_clusters_output", {})
                    .get("path", "data/processed/merged_clusters.parquet")
                }
                if env == "dev"
                else {
                    "table_name": env_config.get("writers", {})
                    .get("merged_clusters_output", {})
                    .get("table_name", "STAGING_CLUSTERING_DB.PROCESSED.MERGED_CLUSTERS")
                },
            }
        ),
    }

    # Combine all resources
    all_resources = {
        **base_resources,
        **io_resources,
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
