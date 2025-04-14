"""Dagster definitions module for the clustering pipeline."""

import os
from types import SimpleNamespace
from typing import Dict, Any

import dagster as dg
import yaml

# Replace the environment resolver with our Hydra-style config loader
from clustering.utils.hydra_config import load_config as hydra_load_config

# -----------------------------------------------------------------------------
# Asset imports
# -----------------------------------------------------------------------------

# Internal preprocessing assets
from clustering.dagster.assets import (
    internal_raw_sales_data,
    internal_product_category_mapping,
    internal_sales_with_categories,
    internal_normalized_sales_data,
    internal_sales_by_category,
    internal_output_sales_table,
)

# External preprocessing assets
from clustering.dagster.assets import (
    external_features_data,
    preprocessed_external_data,
)

# Internal ML assets - feature engineering
from clustering.dagster.assets import (
    internal_fe_raw_data,
    internal_filtered_features,
    internal_imputed_features,
    internal_normalized_data,
    internal_outlier_removed_features,
    internal_dimensionality_reduced_features,
    internal_feature_metadata,
)

# Internal ML assets - model training and analysis
from clustering.dagster.assets import (
    internal_optimal_cluster_counts,
    internal_train_clustering_models,
    internal_save_clustering_models,
    internal_assign_clusters,
    internal_save_cluster_assignments,
    internal_calculate_cluster_metrics,
    internal_generate_cluster_visualizations,
)

# External ML assets - feature engineering 
from clustering.dagster.assets import (
    external_fe_raw_data,
    external_filtered_features,
    external_imputed_features,
    external_normalized_data,
    external_outlier_removed_features,
    external_dimensionality_reduced_features,
    external_feature_metadata,
)

# External ML assets - model training and analysis
from clustering.dagster.assets import (
    external_optimal_cluster_counts,
    external_train_clustering_models,
    external_save_clustering_models,
    external_assign_clusters,
    external_save_cluster_assignments,
    external_calculate_cluster_metrics,
    external_generate_cluster_visualizations,
)

# Merging assets
from clustering.dagster.assets import (
    merged_clusters,
    merged_cluster_assignments,
    optimized_merged_clusters,
    cluster_reassignment,
    save_merged_cluster_assignments,
)

# Resources
from clustering.dagster.resources import data_io, logger_service

# -----------------------------------------------------------------------------
# Job definitions
# -----------------------------------------------------------------------------

# 1. Internal preprocessing job
internal_preprocessing_job = dg.define_asset_job(
    name="internal_preprocessing_job",
    selection=[
        internal_raw_sales_data,
        internal_product_category_mapping,
        internal_sales_with_categories,
        internal_normalized_sales_data,
        internal_sales_by_category,
        internal_output_sales_table,
    ],
    tags={"kind": "internal_preprocessing"},
)

# 2. External preprocessing job
external_preprocessing_job = dg.define_asset_job(
    name="external_preprocessing_job",
    selection=[
        external_features_data,
        preprocessed_external_data,
    ],
    tags={"kind": "external_preprocessing"},
)

# 3. Internal ML job
internal_ml_job = dg.define_asset_job(
    name="internal_ml_job",
    selection=[
        # Feature engineering assets
        internal_fe_raw_data,
        internal_filtered_features,
        internal_imputed_features,
        internal_normalized_data,
        internal_outlier_removed_features,
        internal_dimensionality_reduced_features,
        internal_feature_metadata,
        # Model training assets
        internal_optimal_cluster_counts,
        internal_train_clustering_models,
        internal_save_clustering_models,
        # Cluster assignment
        internal_assign_clusters,
        internal_save_cluster_assignments,
        # Cluster analysis
        internal_calculate_cluster_metrics,
        internal_generate_cluster_visualizations,
    ],
    tags={"kind": "internal_ml"},
)

# 4. External ML job
external_ml_job = dg.define_asset_job(
    name="external_ml_job",
    selection=[
        # Feature engineering assets
        external_fe_raw_data,
        external_filtered_features,
        external_imputed_features,
        external_normalized_data,
        external_outlier_removed_features,
        external_dimensionality_reduced_features,
        external_feature_metadata,
        # Model training assets
        external_optimal_cluster_counts,
        external_train_clustering_models,
        external_save_clustering_models,
        # Cluster assignment
        external_assign_clusters,
        external_save_cluster_assignments,
        # Cluster analysis
        external_calculate_cluster_metrics,
        external_generate_cluster_visualizations,
    ],
    tags={"kind": "external_ml"},
)

# 5. Merging job
merging_job = dg.define_asset_job(
    name="merging_job",
    selection=[
        # Merging assets
        merged_clusters,
        merged_cluster_assignments,
        optimized_merged_clusters,
        cluster_reassignment,
        save_merged_cluster_assignments,
    ],
    tags={"kind": "merging"},
)

# 6. Full pipeline job (combining all jobs)
full_pipeline_job = dg.define_asset_job(
    name="full_pipeline_job",
    selection=[
        # Internal preprocessing
        internal_raw_sales_data,
        internal_product_category_mapping,
        internal_sales_with_categories,
        internal_normalized_sales_data,
        internal_sales_by_category,
        internal_output_sales_table,
        # Internal feature engineering
        internal_fe_raw_data,
        internal_filtered_features,
        internal_imputed_features,
        internal_normalized_data,
        internal_outlier_removed_features,
        internal_dimensionality_reduced_features,
        internal_feature_metadata,
        # Internal model training
        internal_optimal_cluster_counts,
        internal_train_clustering_models,
        internal_save_clustering_models,
        # Internal cluster assignment
        internal_assign_clusters,
        internal_save_cluster_assignments,
        # Internal cluster analysis
        internal_calculate_cluster_metrics,
        internal_generate_cluster_visualizations,
        # External preprocessing
        external_features_data,
        preprocessed_external_data,
        # External feature engineering
        external_fe_raw_data,
        external_filtered_features,
        external_imputed_features,
        external_normalized_data,
        external_outlier_removed_features,
        external_dimensionality_reduced_features,
        external_feature_metadata,
        # External model training
        external_optimal_cluster_counts,
        external_train_clustering_models,
        external_save_clustering_models,
        # External cluster assignment
        external_assign_clusters,
        external_save_cluster_assignments,
        # External cluster analysis
        external_calculate_cluster_metrics,
        external_generate_cluster_visualizations,
        # Merging
        merged_clusters,
        merged_cluster_assignments,
        optimized_merged_clusters,
        cluster_reassignment,
        save_merged_cluster_assignments,
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
    }
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
    except Exception as e:
        print(f"ERROR loading config from {config_path}: {str(e)}")
        # Provide default configuration to prevent errors
        config_data = {
            "job_params": {},
            "logger": {"level": "INFO", "sink": f"logs/dagster_{env}.log"},
            "readers": {},
            "writers": {}
        }

    return config_data

# -----------------------------------------------------------------------------
# Resource definitions
# -----------------------------------------------------------------------------

def get_resources_by_env(env: str = "dev") -> dict[str, dg.ResourceDefinition]:
    """Get resource definitions based on environment.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary of resource definitions
    """
    # Load configuration
    config_data = load_config(env)

    # Extract configuration sections
    job_params = config_data.get("job_params", {})
    logger_config = config_data.get("logger", {})
    readers_config = config_data.get("readers", {})
    writers_config = config_data.get("writers", {})

    # Create config object
    params = SimpleNamespace(**job_params)
    params.env = env  # Add environment name

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
        "logger": logger_service.configured({
            "sink": logger_config.get("sink", f"logs/dagster_{env}.log"),
            "level": logger_config.get("level", "INFO"),
        }),
        
        # Data readers
        "internal_ns_sales": data_io.data_reader.configured(
            readers_config.get("internal_ns_sales", {})
        ),
        "internal_ns_map": data_io.data_reader.configured(
            readers_config.get("internal_ns_map", {})
        ),
        "output_sales_reader": data_io.data_reader.configured(
            readers_config.get("output_sales", {})
        ),
        "external_data_reader": data_io.data_reader.configured(
            readers_config.get("external_data_source", {})
        ),
        "input_external_placerai_reader": data_io.data_reader.configured(
            readers_config.get("external_placerai", {})
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

def create_definitions(env: str = "dev") -> dg.Definitions:
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
        # Preprocessing assets - Internal
        internal_raw_sales_data,
        internal_product_category_mapping,
        internal_sales_with_categories,
        internal_normalized_sales_data,
        internal_sales_by_category,
        internal_output_sales_table,
        
        # Preprocessing assets - External
        external_features_data,
        preprocessed_external_data,
        
        # Feature engineering assets - Internal
        internal_fe_raw_data,
        internal_filtered_features,
        internal_imputed_features,
        internal_normalized_data,
        internal_outlier_removed_features,
        internal_dimensionality_reduced_features,
        internal_feature_metadata,
        
        # Feature engineering assets - External
        external_fe_raw_data,
        external_filtered_features,
        external_imputed_features,
        external_normalized_data,
        external_outlier_removed_features,
        external_dimensionality_reduced_features,
        external_feature_metadata,
        
        # Model training - Internal
        internal_optimal_cluster_counts,
        internal_train_clustering_models,
        internal_save_clustering_models,
        
        # Model training - External
        external_optimal_cluster_counts,
        external_train_clustering_models,
        external_save_clustering_models,
        
        # Cluster assignment - Internal
        internal_assign_clusters,
        internal_save_cluster_assignments,
        
        # Cluster assignment - External
        external_assign_clusters,
        external_save_cluster_assignments,
        
        # Cluster analysis - Internal
        internal_calculate_cluster_metrics,
        internal_generate_cluster_visualizations,
        
        # Cluster analysis - External
        external_calculate_cluster_metrics,
        external_generate_cluster_visualizations,
        
        # Merging assets
        merged_clusters,
        merged_cluster_assignments,
        optimized_merged_clusters,
        cluster_reassignment,
        save_merged_cluster_assignments,
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
defs = create_definitions(env="dev")

def get_definitions():
    """Get the definitions object.
    
    Returns:
        Dagster definitions object for the default (dev) environment
    """
    return defs

# Export symbols
__all__ = ["create_definitions", "get_definitions", "defs"]