# Pipeline Package

Dagster pipeline for clustering data.

## Package Structure

The package follows the recommended `src` layout:

```
pipeline/
├── src/
│   └── pipeline/      # Actual package code
│       ├── assets/
│       ├── resources/
│       ├── jobs/
│       ├── __init__.py
│       ├── app.py
│       └── definitions.py
├── pyproject.toml    # Project configuration
└── README.md
```

## Installation

```bash
cd pipeline
uv pip install -e .
```

## Usage

This package is typically used through the CLI:

```bash
# Run the pipeline
clustering run full_pipeline_job

# Start the Dagster UI
clustering dashboard
```

You can also import the pipeline directly in Python:

```python
from pipeline.app import run_job, run_app

# Run a job
run_job(env="dev")

# Start the Dagster UI
run_app(host="localhost", port=3000, env="dev")
```

## Components

- `pipeline.assets` - Dagster assets (internal, external, merging)
- `pipeline.jobs` - Job definitions
- `pipeline.resources` - Resource definitions
- `pipeline.definitions` - Main definitions file 