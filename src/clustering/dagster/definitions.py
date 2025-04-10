"""Dagster definitions module for the clustering pipeline."""

import os
from types import SimpleNamespace

import dagster as dg
import yaml

# Import all assets directly
from clustering.dagster.assets import (
    cluster_assignments,
    cluster_metrics,
    cluster_visualizations,
    dimensionality_reduced_features,
    fe_raw_data,
    feature_metadata,
    filtered_features,
    imputed_features,
    normalized_data,
    normalized_sales_data,
    optimal_cluster_counts,
    outlier_removed_features,
    output_sales_table,
    persisted_cluster_assignments,
    product_category_mapping,
    raw_sales_data,
    sales_by_category,
    sales_with_categories,
    saved_clustering_models,
    trained_clustering_models,
)
from clustering.dagster.resources import alerts_service, data_io, logger_service


# Define resources with a simple, direct approach
def get_resources_by_env(env: str = "dev") -> dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    # Load configuration directly
    config_path = os.path.join(os.path.dirname(__file__), "resources", "configs", f"{env}.yml")
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Extract job parameters
    job_params = config_data.get("job_params", {})

    # Create config object directly
    params = SimpleNamespace(**job_params)
    params.env = env  # Add environment name

    # Extract other resource configurations
    logger_config = config_data.get("logger", {})
    alerts_config = config_data.get("alerts", {})
    readers_config = config_data.get("readers", {})
    writers_config = config_data.get("writers", {})

    # Create the params resource once
    params_resource = dg.resource(lambda: params)()

    # Create all resources in a flat structure
    resources = {
        # Core resources
        "io_manager": dg.FilesystemIOManager(base_dir="storage"),
        # Use the same params resource for both job_params and config
        # Some assets look for "job_params" while others look for "config"
        "job_params": params_resource,
        "config": params_resource,
        # Logger
        "logger": logger_service.configured(
            {
                "sink": logger_config.get("sink", f"logs/dagster_{env}.log"),
                "level": logger_config.get("level", "INFO"),
            }
        ),
        # Alerts
        "alerts": alerts_service.configured(
            {
                "enabled": alerts_config.get("enabled", True),
                "threshold": alerts_config.get("threshold", "WARNING"),
                "slack_webhook": alerts_config.get("slack_webhook", None),
            }
        ),
        # Data readers
        "internal_ns_sales": data_io.data_reader.configured(
            readers_config.get("internal_ns_sales", {})
        ),
        "internal_ns_map": data_io.data_reader.configured(
            readers_config.get("internal_ns_map", {})
        ),
        # Feature engineering reader
        "output_sales_reader": data_io.data_reader.configured(
            readers_config.get("output_sales", {})
        ),
        # Data writers
        "output_sales_writer": data_io.data_writer.configured(
            writers_config.get("internal_sales_output", {})
        ),
        "output_sales_percent_writer": data_io.data_writer.configured(
            writers_config.get("internal_sales_percent_output", {})
        ),
        "output_clusters_writer": data_io.data_writer.configured(
            writers_config.get("internal_clusters_output", {})
        ),
        # Model output writer
        "model_output": data_io.data_writer.configured(writers_config.get("model_output", {})),
        # Cluster assignments writer
        "cluster_assignments": data_io.data_writer.configured(
            writers_config.get("cluster_assignments", {})
        ),
    }

    return resources


# Define all jobs directly in definitions.py

# Define internal preprocessing job
internal_preprocessing_job = dg.define_asset_job(
    name="internal_preprocessing_job",
    selection=[
        raw_sales_data,
        product_category_mapping,
        sales_with_categories,
        normalized_sales_data,
        sales_by_category,
        output_sales_table,
    ],
    tags={"kind": "internal_preprocessing"},
)

# Define internal ML job directly
internal_clustering_job = dg.define_asset_job(
    name="internal_clustering_job",
    selection=[
        # Feature engineering assets
        fe_raw_data,
        filtered_features,
        imputed_features,
        normalized_data,
        outlier_removed_features,
        dimensionality_reduced_features,
        feature_metadata,
        # Model training assets
        optimal_cluster_counts,
        trained_clustering_models,
        saved_clustering_models,
        # Cluster prediction
        cluster_assignments,
        persisted_cluster_assignments,
        # Evaluation assets
        cluster_metrics,
        cluster_visualizations,
    ],
    tags={"kind": "internal_ml"},
)

# Define full pipeline job
full_pipeline_job = dg.define_asset_job(
    name="full_pipeline_job",
    selection=[
        raw_sales_data,
        product_category_mapping,
        sales_with_categories,
        normalized_sales_data,
        sales_by_category,
        output_sales_table,
        # Feature engineering
        fe_raw_data,
        filtered_features,
        imputed_features,
        normalized_data,
        outlier_removed_features,
        dimensionality_reduced_features,
        feature_metadata,
        # Model training
        optimal_cluster_counts,
        trained_clustering_models,
        saved_clustering_models,
        # Cluster prediction
        cluster_assignments,
        persisted_cluster_assignments,
        # Model evaluation
        cluster_metrics,
        cluster_visualizations,
    ],
    tags={"kind": "full_pipeline"},
)


# Create definitions
def create_definitions(env: str = "dev") -> dg.Definitions:
    """Create Dagster definitions.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dagster definitions
    """
    resources = get_resources_by_env(env)

    # Include all assets in one place
    assets = [
        # Preprocessing assets - Internal
        raw_sales_data,
        product_category_mapping,
        sales_with_categories,
        normalized_sales_data,
        sales_by_category,
        output_sales_table,
        # Feature engineering assets
        fe_raw_data,
        filtered_features,
        imputed_features,
        normalized_data,
        outlier_removed_features,
        dimensionality_reduced_features,
        feature_metadata,
        # Model training
        optimal_cluster_counts,
        trained_clustering_models,
        saved_clustering_models,
        # Cluster prediction
        cluster_assignments,
        persisted_cluster_assignments,
        # Model evaluation
        cluster_metrics,
        cluster_visualizations,
    ]

    # Create and return definitions with all jobs defined directly
    return dg.Definitions(
        assets=assets,
        resources=resources,
        jobs=[
            internal_preprocessing_job,
            internal_clustering_job,
            full_pipeline_job,
        ],
    )


# Create default definitions with dev environment
defs = create_definitions(env="dev")
