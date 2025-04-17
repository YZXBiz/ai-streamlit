# Clustering Dashboard

A Streamlit dashboard for visualizing and exploring cluster assignments from the Dagster clustering pipeline.

## Features

- **Asset Selection**: By default focuses on visualizing merging assets, with option for Excel file uploads
- **Cluster Distribution**: View the distribution of stores across clusters
- **Feature Analysis**: Explore relationships between features and clusters
- **Cluster Comparison**: Compare cluster assignments before and after optimization
- **Visual Data Explorer**: Drag-and-drop interface for interactive data visualization using PyGWalker
- **Environment Variable Configuration**: Override settings with environment variables
- **Type-Safe Settings**: Pydantic-based settings with validation

## Structure

```
dashboard/
├── __main__.py              # CLI entry point for running the dashboard
├── app.py                   # Main Streamlit application
├── components/              # Reusable dashboard components
│   ├── data_loader.py       # Data loading utilities
│   ├── cluster_view.py      # Cluster visualization components
│   ├── feature_explorer.py  # Feature exploration components
│   ├── feature_selector.py  # UI for selecting features to visualize
│   ├── visualization.py     # Generic visualization components
│   ├── pygwalker_view.py    # PyGWalker integration for visual data exploration
│   └── visualizer.py        # Plotting utilities
├── config/                  # Dashboard configuration
│   └── settings.py          # Pydantic-based settings management
├── utils/                   # Utility functions
│   └── __init__.py          # Color scales and helper functions
└── README.md                # Dashboard documentation
```

## Running the Dashboard

To run the dashboard:

```bash
# From the project root directory
cd src
streamlit run -m clustering.dashboard

# Or with uv (recommended)
cd src
uv run -m clustering.dashboard
```

Or use the Makefile target:

```bash
make dashboard
```

## Visual Data Explorer with PyGWalker

The dashboard includes integration with PyGWalker, which provides a drag-and-drop interface for creating visualizations without writing code.

### Features
- **No-code visualization**: Create charts by simply dragging and dropping fields
- **Multiple chart types**: Bar charts, scatter plots, line charts, heatmaps and more
- **Interactive filtering**: Filter and transform your data visually
- **Export capabilities**: Export chart specifications to reuse later

### Installation
PyGWalker is not included in the base requirements. To use this feature:

```bash
pip install pygwalker
```

Or with uv:
```bash
uv pip install pygwalker
```

After installation, restart the dashboard to access the PyGWalker tab.

## Code Style & Formatting

This project uses ruff for code formatting and linting. To format your code:

```bash
# Format all Python files in the project
ruff format .

# Format a specific file
ruff format clustering/dashboard/app.py

# Check formatting without making changes
ruff format --check .
```

To run linting checks with import sorting:

```bash
# Run linting with import sorting
ruff check --select I --fix
ruff format
```

## Configuration

The dashboard can be configured using:

1. **Environment Variables**: Override any setting with `DASHBOARD_` prefix
   ```bash
   # Examples
   export DASHBOARD_DASHBOARD_TITLE="My Custom Dashboard"
   export DASHBOARD_THEME__PRIMARY_COLOR="#FF5722"
   export DASHBOARD_MAX_FEATURES_TO_DISPLAY=30
   ```

2. **dotenv File**: Create a `.env` file in the project root with configuration
   ```
   DASHBOARD_DASHBOARD_TITLE=My Custom Dashboard
   DASHBOARD_THEME__PRIMARY_COLOR=#FF5722
   ```

3. **Programmatic Settings**: Modify `config/settings.py` for permanent changes

The configuration uses Pydantic Settings for type validation and easy environment variable overrides.

## Available Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| Dashboard Title | `DASHBOARD_DASHBOARD_TITLE` | "Clustering Dashboard" | The title displayed at the top of the dashboard |
| Theme Primary Color | `DASHBOARD_THEME__PRIMARY_COLOR` | "#4CAF50" | Primary color for interactive elements |
| Max Features | `DASHBOARD_MAX_FEATURES_TO_DISPLAY` | 20 | Maximum number of features to display |
| Layout | `DASHBOARD_LAYOUT` | "wide" | Dashboard layout ("wide" or "centered") |
| Storage Path | `DASHBOARD_STORAGE_PATH` | "../storage" | Path to Dagster storage directory |

## Data Sources

The dashboard supports the following data sources:

1. **Merging Assets**: By default, the dashboard loads merging assets from the Dagster pipeline, focused on the final cluster assignments after internal and external data has been combined.

2. **Excel File Upload**: Users can upload their own Excel files for visualization and analysis.

## Extending

### Adding New Visualizations

To add new visualizations:

1. Create a new function in the appropriate component file:
```python
def show_new_visualization(df: pd.DataFrame, cluster_col: str, features: list[str]) -> None:
    """Show a new type of visualization.
    
    Args:
        df: DataFrame with data
        cluster_col: Name of cluster column
        features: List of feature columns
    """
    # Create and display visualization
    # ...
```

2. Import it in `app.py`
3. Add it to the appropriate tab in `render_cluster_analysis()`

### Adding New Data Sources

To add a new data source:

1. Create or modify the appropriate function in `data_loader.py`
2. Update the data source selector UI in `app.py`

## Dependencies

- Python 3.10+
- Streamlit
- Pandas 
- Plotly
- scikit-learn
- Dagster (for asset integration)
- Pydantic Settings (for configuration)

## Contributing

1. Ensure all functions have proper type hints using Python 3.10+ syntax
2. Follow Google docstring format for all functions and classes
3. Run tests with `uv run -m pytest tests/dashboard`
4. Format code with `ruff format` and check types with `mypy` 