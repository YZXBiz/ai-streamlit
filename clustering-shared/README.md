# Clustering Shared

This package contains shared utilities and common functionality used across the store clustering project components.

## Installation

```bash
uv add -e .
```

## Usage

Import the package using the namespace pattern:

```python
import clustering.shared

# Example usage
from clustering.shared.utils import load_config
```

## Structure

The package follows the namespace package pattern:

```
clustering-shared/
└── src/
    └── clustering/      # Namespace package (no __init__.py here)
        └── shared/      # Actual package
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