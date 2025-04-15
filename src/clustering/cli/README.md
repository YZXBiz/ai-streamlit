# Clustering Pipeline CLI

A command-line interface for managing the clustering pipeline, including running jobs, managing sensors, and accessing the web UI.

## Installation

### As a developer

If you're developing this project:

```bash
# Clone the repository
git clone <repository-url>
cd testing-dagster

# Install in development mode
pip install -e .
```

### As a user

```bash
# Install from PyPI
pip install clustering-pipeline

# Or install directly from GitHub
pip install git+https://github.com/username/testing-dagster.git
```

## Usage

The CLI provides several commands for interacting with the Dagster-based clustering pipeline.

### Getting Help

```bash
# Show available commands
uv run -m clustering.cli --help

# Show help for a specific command
uv run -m clustering.cli run --help
```

### Listing Available Jobs

To see all available jobs in the repository:

```bash
uv run -m clustering.cli list_jobs
```

### Running Jobs

To run a specific Dagster job:

```bash
uv run -m clustering.cli run JOB_NAME [OPTIONS]
```

#### Available Jobs

- `internal_preprocessing_job` - Preprocess internal data sources
- `internal_clustering_job` - Run clustering on internal data
- `external_preprocessing_job` - Preprocess external data sources
- `external_clustering_job` - Run clustering on external data 
- `merging_job` - Merge internal and external clustering results
- `full_pipeline_job` - Run the complete pipeline from preprocessing to merging

#### Options

- `--env [dev|staging|prod]` - Environment to use (default: dev)
- `--tags KEY=VALUE` - Add tags to the run (can be specified multiple times)
- `--verbose, -v` - Show verbose output

#### Examples

```bash
# Run the full pipeline in development environment
uv run -m clustering.cli run full_pipeline_job --env dev

# Run internal preprocessing with custom tags
uv run -m clustering.cli run internal_preprocessing_job --tags dataset=customers --tags version=1.0 --verbose
```

### Web UI

Launch the Dagster web UI for monitoring and managing the pipeline:

```bash
uv run -m clustering.cli ui [OPTIONS]
```

#### Options

- `--host TEXT` - Host to bind to (default: localhost)
- `--port INTEGER` - Port to bind to (default: 3000)
- `--env [dev|staging|prod]` - Environment to use (default: dev)

#### Example

```bash
# Start the UI on port 3000
uv run -m clustering.cli ui

# Start the UI on a custom host and port
uv run -m clustering.cli ui --host 0.0.0.0 --port 8080 --env prod
```

### Minimal Demo

Run a minimal example with SQL engine for demonstration purposes:

```bash
uv run -m clustering.cli minimal [OPTIONS]
```

#### Options

- `--host TEXT` - Host to bind to (default: localhost)
- `--port INTEGER` - Port to bind to (default: 3000)

#### Example

```bash
# Run the minimal demo
uv run -m clustering.cli minimal
```

## Environment Configuration

The CLI supports different environments (dev, staging, prod) and will load environment variables from:

1. `.env.{env}` file in the project root (e.g., `.env.dev`)
2. `.env` file in the project root (fallback)

## Common Workflows

### Development Testing

```bash
# List available jobs
uv run -m clustering.cli list_jobs

# Run minimal demo to check functionality
uv run -m clustering.cli minimal

# Run a specific job with verbose output
uv run -m clustering.cli run internal_preprocessing_job --verbose
```

### Production Deployment

```bash
# Run the full pipeline in production environment
uv run -m clustering.cli run full_pipeline_job --env prod --tags version=2.0

# Launch the UI in production mode
uv run -m clustering.cli ui --env prod --host 0.0.0.0
``` 