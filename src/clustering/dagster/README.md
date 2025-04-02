# Dagster Implementation for Clustering Project

This directory contains the Dagster implementation for the clustering project, providing data orchestration, scheduling, and monitoring capabilities.

## Overview

The Dagster implementation organizes the existing clustering project into a structured pipeline with well-defined assets, resources, and schedules. It maintains the same business logic while providing the benefits of Dagster's asset-based paradigm.

## Project Structure

```
src/clustering/dagster/
├── assets/                  # Dagster assets (converted from jobs)
│   ├── preprocessing/       # Preprocessing assets
│   │   ├── internal.py      # Internal preprocessing assets
│   │   └── external.py      # External preprocessing assets
│   ├── clustering/          # Clustering assets
│   │   ├── internal.py      # Internal clustering assets
│   │   └── external.py      # External clustering assets
│   └── merging/             # Merging operations as assets
├── resources/               # Dagster resources
│   ├── config.py            # Configuration resources
│   ├── io.py                # I/O service resources
│   └── io_manager.py        # Custom IO manager
├── schedules/               # Dagster schedules
│   └── __init__.py          # Schedule definitions
├── definitions/             # Dagster repository definition
│   └── __init__.py          # Main definitions for jobs and assets
└── README.md                # This file
```

## Key Components

1. **Assets**: These represent data objects in your pipeline. They are sourced from previous job implementations but restructured to fit Dagster's asset paradigm.
   - Internal/External Preprocessing Assets
   - Internal/External Clustering Assets
   - Merging Assets

2. **Resources**: Resources provide access to external systems and configuration. They include:
   - Configuration management (loading from YAML files)
   - Logging and alerts
   - I/O operations for reading/writing data

3. **IO Manager**: Custom IO manager for handling storage of different data types, including DataFrames and dictionaries.

4. **Schedules**: Automated execution schedules:
   - Daily: Internal clustering pipeline
   - Weekly: External clustering pipeline
   - Monthly: Full pipeline (preprocessing, clustering, and merging)

5. **Definitions**: Main entry point that ties together assets, resources, and schedules.

## Running the Dagster UI

To start the Dagster UI:

```bash
python -m clustering.dagster_app --env dev
```

This will start the Dagster UI on http://127.0.0.1:3000

## Environment Support

The implementation supports different environments (dev, staging, prod) through environment-specific configuration:

```bash
# Development environment
python -m clustering.dagster_app --env dev

# Staging environment
python -m clustering.dagster_app --env staging

# Production environment
python -m clustering.dagster_app --env prod
```

## Asset Execution

You can run individual assets or entire jobs from the Dagster UI or using the CLI:

```bash
# Run the internal preprocessing job
dagster job execute -f clustering.dagster:defs internal_preprocessing_job

# Run the internal clustering job
dagster job execute -f clustering.dagster:defs internal_clustering_job

# Run the external preprocessing job
dagster job execute -f clustering.dagster:defs external_preprocessing_job

# Run the external clustering job
dagster job execute -f clustering.dagster:defs external_clustering_job

# Run the merging job
dagster job execute -f clustering.dagster:defs merging_job

# Run the full pipeline
dagster job execute -f clustering.dagster:defs full_pipeline_job
```

## Available Jobs

1. **internal_preprocessing_job**: Preprocess internal sales and need state data
2. **internal_clustering_job**: Run clustering on preprocessed internal data
3. **external_preprocessing_job**: Preprocess external features data
4. **external_clustering_job**: Run clustering on preprocessed external data
5. **merging_job**: Merge internal and external clustering results
6. **full_pipeline_job**: Run the entire pipeline end-to-end

## Schedules

1. **daily_internal_clustering_schedule**: Daily run of internal clustering (midnight UTC)
2. **weekly_external_clustering_schedule**: Weekly run of external clustering (Monday at midnight UTC)
3. **monthly_full_pipeline_schedule**: Monthly run of the full pipeline (1st of month at midnight UTC)

## Migration Approach

This implementation preserves the existing business logic from the original jobs while adapting them to Dagster's asset-based approach. The original code structure (core, io, utils) is maintained, with the Dagster layer added as a thin orchestration layer. 