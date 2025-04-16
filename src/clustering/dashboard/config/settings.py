"""Settings and configuration for the clustering dashboard."""

import os
from typing import Dict, List, Any

# Dashboard title and description
DASHBOARD_TITLE = "Clustering Dashboard"
DASHBOARD_SUBTITLE = "Visualize and explore cluster assignments"

# Default assets to show in the dashboard
DEFAULT_ASSETS = [
    "cluster_reassignment",
    "merged_clusters",
    "optimized_merged_clusters"
]

# Dashboard layout and UI settings
LAYOUT = "wide"  # "wide" or "centered"
SIDEBAR_STATE = "expanded"  # "expanded" or "collapsed"
THEME = {
    "primaryColor": "#7792E3",
    "backgroundColor": "#273346",
    "secondaryBackgroundColor": "#1E293B",
    "textColor": "#FFFFFF",
    "font": "sans-serif"
}

# Feature settings
MAX_FEATURES_TO_DISPLAY = 20  # Maximum number of features to display in visualizations

# Paths
STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

# Visualization settings
CHART_HEIGHT = 500  # Default height for charts
COLOR_SCALES = {
    "sequential": "Viridis",
    "diverging": "RdBu",
    "categorical": "Set1"
}

# Map of asset types to their descriptions
ASSET_DESCRIPTIONS: Dict[str, str] = {
    "cluster_reassignment": "Final cluster assignments after small cluster reassignment",
    "merged_clusters": "Combined internal and external clusters before optimization",
    "optimized_merged_clusters": "Dictionary with small and large clusters identified",
    "merged_cluster_assignments": "Mapping of merged cluster assignments with counts",
    "internal_assign_clusters": "Internal data cluster assignments",
    "external_assign_clusters": "External data cluster assignments",
    "internal_train_clustering_models": "Trained clustering models for internal data",
    "external_train_clustering_models": "Trained clustering models for external data"
}

# Default visualizations to show for each asset type
DEFAULT_VISUALIZATIONS: Dict[str, List[str]] = {
    "cluster_reassignment": ["cluster_distribution", "feature_scatter", "comparison"],
    "merged_clusters": ["cluster_distribution", "feature_scatter", "parallel_coordinates"],
    "internal_assign_clusters": ["cluster_distribution", "feature_scatter", "3d_scatter"],
    "external_assign_clusters": ["cluster_distribution", "feature_scatter", "3d_scatter"]
} 