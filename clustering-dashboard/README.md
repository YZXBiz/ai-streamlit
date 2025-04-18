# Clustering Dashboard

This package contains the web dashboard for visualizing and interacting with the store clustering project.

## Installation

```bash
uv add -e .
```

## Usage

Import the package using the namespace pattern:

```python
import clustering.dashboard

# Example usage
from clustering.dashboard.app import run_dashboard
```

## Structure

The package follows the namespace package pattern:

```
clustering-dashboard/
└── src/
    └── clustering/      # Namespace package (no __init__.py here)
        └── dashboard/   # Actual package
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

Run the dashboard:
```bash
uv run streamlit run src/clustering/dashboard/app.py
``` 