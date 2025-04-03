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

# Dagster Implementation - Quick Start Guide

This guide explains how to get started with the Dagster implementation of the clustering pipeline. It focuses on a step-by-step approach from simple testing to full production deployment.

## Setup with UV

1. **Create a virtual environment using UV**:

   ```bash
   uv venv
   # Activate the environment
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   uv sync --all-extras --dev
   ```

## Running the Minimal Example

```bash
# Start the Dagster UI with the minimal example
python -m clustering minimal
```

This minimal example demonstrates:

- A simple 2-asset pipeline
- Sample data generation
- SQL transformation using the DuckDB engine

## Testing SQL Transformations

To test new SQL logic before adding it to your codebase:

1. **Create a test script**:

   ```python
   import polars as pl
   from clustering.core.sql_engine import DuckDB, SQL

   # Sample test data
   test_data = pl.DataFrame({
       "SKU_NBR": ["SKU001", "SKU002", "SKU003"],
       "STORE_NBR": ["S001", "S002", "S003"],
       "TOTAL_SALES": [100, 200, 300]
   })

   # Your SQL transformation
   sql_obj = SQL(
       """
       SELECT
           "SKU_NBR",
           "STORE_NBR",
           "TOTAL_SALES",
           "TOTAL_SALES" * 0.1 AS "TAX_AMOUNT"
       FROM $data
       """,
       bindings={"data": test_data}
   )

   # Execute and print results
   db = DuckDB()
   try:
       result = db.query(sql_obj)
       print(result)
   finally:
       db.close()
   ```

2. **Run the script**:

   ```bash
   python your_test_script.py
   ```

3. **Once validated, add to `sql_templates.py`**:
   ```python
   def calculate_tax(data_df: pl.DataFrame) -> SQL:
       """Create SQL to calculate tax amount.

       Args:
           data_df: Input dataframe with sales data

       Returns:
           SQL object for calculating tax amounts
       """
       return SQL(
           """
           SELECT
               "SKU_NBR",
               "STORE_NBR",
               "TOTAL_SALES",
               "TOTAL_SALES" * 0.1 AS "TAX_AMOUNT"
           FROM $data
           """,
           bindings={"data": data_df}
       )
   ```

## Running the Full Pipeline

Once you understand the minimal example, you can run the full pipeline:

```bash
# Start the full Dagster UI
python -m clustering ui
```

## Available Commands

The clustering CLI provides several commands:

- `python -m clustering minimal`: Run the minimal example UI
- `python -m clustering ui`: Run the full Dagster UI
- `python -m clustering run JOB_NAME`: Run a specific job and exit

## Development Workflow

1. **Test new SQL logic** in isolation
2. **Add validated SQL templates** to `clustering.core.sql_templates`
3. **Create Dagster assets** using the templates
4. **Test assets** individually
5. **Compose assets into jobs** for production use

## Troubleshooting

If you encounter issues:

1. **Verify Python version compatibility**:

   ```bash
   python --version
   ```

   Ensure you're using Python 3.11 as some dependencies like PyCaret require it.

2. **Verify dependencies**:

   ```bash
   uv list
   ```

3. **Check path configuration**:
   ```bash
   python -c "import sys; print(sys.path)"
   ```
   Ensure `src` is in the Python path.
