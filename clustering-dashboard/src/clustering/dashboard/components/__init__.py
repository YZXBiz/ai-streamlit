"""Components module for clustering dashboard."""

from clustering.dashboard.components.correlation_matrix import (
    create_correlation_matrix,
    display_correlation_matrix,
)
from clustering.dashboard.components.data_summary import (
    display_complete_data_summary,
    display_dataset_metrics,
    display_datatype_chart,
    display_numeric_stats,
    display_schema_summary,
)
from clustering.dashboard.components.metric_card import metric_card, metric_row
from clustering.dashboard.components.parallel_coordinates import (
    create_parallel_coordinates,
    display_parallel_coordinates_with_controls,
)
from clustering.dashboard.components.pygwalker_view import get_pyg_renderer
from clustering.dashboard.components.scatter_plot import (
    create_scatter_plot,
    display_scatter_plot_with_controls,
)
