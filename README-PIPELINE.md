# Dagster Clustering Pipeline

This document explains how to set up and run the Dagster clustering pipeline for analyzing internal and external sales data.

## Prerequisites

- Python 3.10 or higher
- `uv` for dependency management
- Required Python packages (can be installed with `uv install`)

## Project Structure

The project is organized as follows:

```
.
├── configs/                  # Configuration files
├── data/                     # Data directory
│   └── raw/                  # Raw input data
│       ├── internal_sales.parquet
│       ├── need_state.csv
│       └── external_sales.parquet
├── outputs/                  # Output directory for processed data
├── scripts/                  # Utility scripts
│   ├── generate_sample_data.py  # Script to generate sample data
│   └── run_pipeline.py       # Script to run the pipeline
└── src/                      # Source code
    └── clustering/           # Clustering package
        └── dagster/          # Dagster definitions
```

## Setup

1. Generate sample data (if needed):

   ```bash
   python scripts/generate_sample_data.py
   ```

   This will create the following files:

   - `data/raw/internal_sales.parquet`
   - `data/raw/need_state.csv`
   - `data/raw/external_sales.parquet`

2. Ensure all required directories exist:

   ```bash
   mkdir -p data/raw outputs
   ```

## Running the Pipeline

There are two ways to run the pipeline:

### Option 1: Using the Dagster UI

Start the Dagster UI server:

```bash
dagster dev -f src/clustering/dagster/definitions.py
```

Then open your browser at http://localhost:3000 to:

- View the pipeline structure
- Launch jobs manually
- Monitor job execution
- View asset materializations

### Option 2: Using the Command Line Script

Run the pipeline from the command line:

```bash
# Run the full pipeline
python scripts/run_pipeline.py --job full_pipeline_job --env dev

# Run specific jobs
python scripts/run_pipeline.py --job internal_preprocessing_job --env dev
python scripts/run_pipeline.py --job internal_clustering_job --env dev
python scripts/run_pipeline.py --job external_preprocessing_job --env dev
python scripts/run_pipeline.py --job external_clustering_job --env dev
python scripts/run_pipeline.py --job merging_job --env dev

# Add custom tags
python scripts/run_pipeline.py --job full_pipeline_job --tag run_type=test --tag version=1.0.0
```

## Pipeline Components

The pipeline consists of the following jobs:

1. **Internal Preprocessing Job**: Preprocesses internal sales data

   - Reads internal sales and need state data
   - Merges data and extracts categories
   - Produces preprocessed sales data

2. **Internal Clustering Job**: Clusters internal data

   - Normalizes data
   - Performs clustering
   - Evaluates clusters
   - Generates clustering output

3. **External Preprocessing Job**: Preprocesses external sales data

   - Reads external features data
   - Produces preprocessed external data

4. **External Clustering Job**: Clusters external data

   - Builds clustering model
   - Performs clustering
   - Evaluates clusters
   - Generates clustering output

5. **Merging Job**: Merges internal and external clusters

   - Combines clusters from both sources
   - Generates final merged output

6. **Full Pipeline Job**: Runs all of the above jobs in sequence

## Configuration

The pipeline uses configuration files in YAML format:

- Environment configs: `src/clustering/dagster/resources/configs/{env}.yml`
- Job configs: `src/clustering/dagster/resources/configs/job_configs/{job}.yml`

You can modify these files to adjust pipeline behavior for different environments and job parameters.

## Output

The pipeline produces various outputs:

- Preprocessed data files in the `outputs/` directory
- A DuckDB database with asset tables in `outputs/clustering_dev.duckdb`
- Log files in the specified log directory

## Troubleshooting

If you encounter issues:

1. Check the log files for detailed error messages
2. Ensure all required data files exist
3. Verify that configuration files are properly set up
4. Make sure all dependencies are correctly installed
