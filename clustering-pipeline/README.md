# Clustering Pipeline

This package contains the Dagster pipeline for data processing and machine learning tasks related to the store clustering project.

## Installation

```bash
uv add -e .
```

## Usage

Import the package using the namespace pattern:

```python
import clustering.pipeline

# Example usage
from clustering.pipeline.definitions import defs
```

## Structure

The package follows the namespace package pattern:

```
clustering-pipeline/
└── src/
    └── clustering/      # Namespace package (no __init__.py here)
        └── pipeline/    # Actual package
            └── __init__.py
            └── ...
```

## Development

Run tests:
```bash
uv run pytest tests/
```

Run linting:
```bash
uv run ruff check .
``` 