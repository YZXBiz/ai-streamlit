"""Dagster definitions for the clustering pipeline."""

from typing import Dict

import dagster as dg

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
from clustering.dagster.resources import alerts_service, clustering_config, clustering_io_manager, logger_service
from clustering.dagster.schedules import (
    daily_internal_clustering_schedule,
    monthly_full_pipeline_schedule,
    weekly_external_clustering_schedule,
)


# Define resource config by environment
def get_resources_by_env(env: str = "dev") -> Dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    base_resources = {
        # IO Manager
        "io_manager": clustering_io_manager.configured({"base_dir": f"outputs/dagster_storage/{env}"}),
        # Config
        "config": clustering_config.configured({"env": env, "config_path": "configs/internal_clustering.yml"}),
        # Logger
        "logger": logger_service.configured({"sink": f"logs/dagster_{env}.log", "level": "INFO"}),
        # Alerts
        "alerts": alerts_service.configured({"enabled": True, "threshold": "WARNING"}),
    }

    return base_resources


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


# Default definitions for dev environment
defs = create_definitions("dev")
