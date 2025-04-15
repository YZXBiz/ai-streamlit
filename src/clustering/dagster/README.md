# Dagster Pipeline Implementation

This directory contains the Dagster implementation of the store clustering pipeline.

## Directory Structure

```
dagster/
├── assets/                  # Asset definitions
│   ├── external/            # External data assets
│   ├── internal/            # Internal data assets
│   ├── merging/             # Cluster merging assets
│   └── common/              # Shared assets
├── resources/               # Resource definitions
│   ├── configs/             # Configuration files
│   ├── io/                  # I/O resource implementations
│   └── utils/               # Utility resources
├── definitions.py           # Main pipeline definitions
├── app.py                   # Dagster app configuration
└── __init__.py              # Package initialization
```

## Key Components

### Assets

Assets are the main building blocks of the pipeline, representing data objects and their transformations:

- **Internal Processing Assets**: Process raw sales data and product mappings
- **Feature Engineering Assets**: Apply normalization, imputation, and dimensionality reduction
- **Model Training Assets**: Train clustering models and determine optimal cluster counts
- **External Data Assets**: Process external data sources
- **Merging Assets**: Combine internal and external clustering results

### Resources

Resources provide the operational components needed by assets:

- **Readers/Writers**: Handle data I/O with configurable paths
- **Configuration**: Environment-specific settings
- **Logging**: Structured logging facilities

### Configuration

The pipeline uses a configuration system based on YAML files with environment variable substitution:

```
resources/configs/
├── dev.yml                 # Development configuration
├── staging.yml             # Staging configuration
└── prod.yml                # Production configuration
```

Configuration includes:
- Feature engineering parameters
- Model training settings
- I/O paths with environment variable support
- Logging configuration

## Running the Pipeline

The pipeline is designed to be run using the Makefile commands:

```bash
# Start the Dagster UI
make dev

# Run specific pipeline components
make run-internal-preprocessing
make run-internal-ml
make run-external-preprocessing
make run-external-ml
make run-merging

# Run the full pipeline with a specific environment
make run-full ENV=prod
```

## Environment Variables

The pipeline respects the following environment variables for path configuration:

- `DATA_DIR`: Base directory for all data
- `INTERNAL_DATA_DIR`: Directory for internal data
- `EXTERNAL_DATA_DIR`: Directory for external data
- `MERGING_DATA_DIR`: Directory for merged data
- `DAGSTER_HOME`: Dagster home directory

## Development

When extending the pipeline:

1. Define new assets in the appropriate module under `assets/`
2. Register assets in `definitions.py`
3. Update configuration in `resources/configs/` as needed
4. Test changes by running the affected pipeline components 