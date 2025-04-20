"""Utility functions for the clustering dashboard."""

import plotly.colors as pc

# Import visualization utilities for easier access
from clustering.dashboard.utils.visualization_utils import (
    apply_filters,
    cached_dataframe,
    create_figure_with_dropdown,
    create_sidebar_filters,
    create_time_series_plot,
    display_code,
    download_dataframe,
    export_dashboard_state,
    format_file_size,
    get_color_scale,
    get_file_info,
    import_dashboard_state,
    load_dataset,
    load_image,
    plot_correlation_matrix,
    plot_missing_values,
    show_dataframe_info,
)

# Export all imported functions
__all__ = [
    "get_color_scale",
    "load_dataset",
    "format_file_size",
    "get_file_info",
    "show_dataframe_info",
    "plot_missing_values",
    "display_code",
    "load_image",
    "download_dataframe",
    "create_figure_with_dropdown",
    "create_sidebar_filters",
    "apply_filters",
    "cached_dataframe",
    "plot_correlation_matrix",
    "create_time_series_plot",
    "export_dashboard_state",
    "import_dashboard_state",
    "VIRIDIS_DARK",
    "DIVERGING_COLORS",
    "QUALITATIVE_COLORS",
]

# Define a dark version of the Viridis colorscale for better contrast
VIRIDIS_DARK = [
    [0.0, "#440154"],
    [0.1111111111111111, "#482878"],
    [0.2222222222222222, "#3e4989"],
    [0.3333333333333333, "#31688e"],
    [0.4444444444444444, "#26828e"],
    [0.5555555555555556, "#1f9e89"],
    [0.6666666666666666, "#35b779"],
    [0.7777777777777778, "#6ece58"],
    [0.8888888888888888, "#b5de2b"],
    [1.0, "#fde725"],
]

# Define a diverging color scale for correlation matrices
DIVERGING_COLORS = [
    [0.0, "#1a0c5a"],
    [0.2, "#3e4989"],
    [0.4, "#6b8dbb"],
    [0.5, "#ffffff"],
    [0.6, "#e58368"],
    [0.8, "#c8243c"],
    [1.0, "#7a0403"],
]

# Define a qualitative color scale for categorical data
QUALITATIVE_COLORS = pc.qualitative.Bold

# Note: get_color_scale function has been moved to visualization_utils.py
