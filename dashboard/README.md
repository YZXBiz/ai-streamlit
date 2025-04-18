# Dashboard Package

Streamlit dashboard for visualizing clustering results.

## Package Structure

The package follows the recommended `src` layout:

```
dashboard/
├── src/
│   └── dashboard/      # Actual package code
│       ├── components/
│       ├── config/
│       ├── utils/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py
├── pyproject.toml     # Project configuration
├── Makefile           # Development utilities
└── README.md
```

## Features

- **Asset Selection**: By default focuses on visualizing merging assets, with option for Excel file uploads
- **Cluster Distribution**: View the distribution of stores across clusters
- **Feature Analysis**: Explore relationships between features and clusters
- **Cluster Comparison**: Compare cluster assignments before and after optimization
- **Visual Data Explorer**: Drag-and-drop interface for interactive data visualization using PyGWalker

## Installation

```bash
cd dashboard
uv pip install -e .
```

## Running the Dashboard

Using the Makefile:

```bash
cd dashboard
make run
```

Or directly:

```bash
cd dashboard
uv run -m dashboard
```

## Configuration

The dashboard can be configured using:

1. **Environment Variables**: Override any setting with `DASHBOARD_` prefix
   ```bash
   export DASHBOARD_DASHBOARD_TITLE="My Custom Dashboard"
   export DASHBOARD_THEME__PRIMARY_COLOR="#FF5722"
   ```

2. **dotenv File**: Create a `.env` file in the dashboard directory with configuration

## Development

```bash
# Start the dashboard in development mode (auto-reload)
make dev

# Format code
make format

# Lint code
make lint
``` 