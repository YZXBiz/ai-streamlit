# Shared Package

Shared utilities and infrastructure for the clustering project.

## Package Structure

The package follows the recommended `src` layout:

```
shared/
├── src/
│   └── shared/         # Actual package code
│       └── utils/
│           ├── io/
│           ├── infra/
│           └── schemas/
├── pyproject.toml     # Project configuration
└── README.md
```

## Installation

```bash
cd shared
uv pip install -e .
```

## Usage

This package contains shared utilities used by other packages in the monorepo:

```python
# IO utilities
from shared.utils.io import load_csv, save_csv

# Infrastructure utilities
from shared.utils.infra import Environment

# Schema definitions
from shared.utils.schemas import ProductCategorySchema
```

## Overview

This package contains:

- Core domain models and utilities
- IO utilities for data input/output
- Infrastructure components (settings, logging, configuration)
- Common utilities used across the project

## Components

- `shared.core` - Core domain models and utilities
- `shared.io` - Data input/output utilities
- `shared.infra` - Infrastructure components
- `shared.utils` - Common utilities 