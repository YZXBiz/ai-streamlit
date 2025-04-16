# Clustering Dashboard

A Streamlit dashboard for visualizing and exploring cluster assignments from the Dagster clustering pipeline.

## Features

- **Asset Selection**: Choose any Dagster asset to visualize
- **Cluster Distribution**: View the distribution of stores across clusters
- **Feature Analysis**: Explore relationships between features and clusters
- **Dimension Reduction**: Use PCA and t-SNE for high-dimensional data visualization
- **Cluster Comparison**: Compare cluster assignments before and after optimization

## Structure

```
dashboard/
├── __init__.py
├── app.py                 # Main Streamlit application
├── components/            # Reusable dashboard components
│   ├── __init__.py
│   ├── cluster_view.py    # Cluster visualization components
│   ├── data_loader.py     # Data loading utilities
│   └── feature_explorer.py # Feature exploration components
├── config/                # Dashboard configuration
│   └── settings.py        # Dashboard settings
└── README.md              # Dashboard documentation
```

## Running the Dashboard

To run the dashboard:

```bash
# From the project root directory
cd src
streamlit run -m clustering.dashboard.app
```

Or use the Makefile target:

```bash
make dashboard
```

## Customization

The dashboard is fully customizable through the `config/settings.py` file. You can adjust:

- Default assets to display
- Dashboard layout and theme
- Visualization settings
- Feature display limits

## Extending

To add new visualizations:

1. Create a new function in the appropriate component file
2. Import it in `app.py`
3. Add it to the appropriate tab in `render_cluster_analysis()` 