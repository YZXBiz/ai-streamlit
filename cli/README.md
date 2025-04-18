# CLI Package

Command-line interface for the clustering project.

## Package Structure

The package follows the standard `src` layout:

```
cli/
├── src/
│   └── cli/         # Actual package code
│       ├── __init__.py
│       └── commands.py
├── pyproject.toml   # Project configuration
└── README.md
```

## Installation

```bash
cd cli
uv pip install -e .
```

## Usage

```bash
# Run the CLI
clustering --help
```

## Commands

- `clustering run <job_name> [--env=dev|test|prod]` - Run a pipeline job
- `clustering dashboard [--host=localhost] [--port=3000] [--env=dev|test|prod]` - Start the Dagster dashboard UI

## Examples

```bash
# Run the full pipeline job
clustering run full_pipeline_job --env=dev

# Start the dashboard UI
clustering dashboard --port=3001 --env=prod
``` 