"""Utility functions for the clustering dashboard."""

from typing import Literal

import plotly.colors as pc

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

def get_color_scale(scale_type: Literal["sequential", "diverging", "qualitative"]) -> list:
    """Get a color scale based on the scale type.
    
    Args:
        scale_type: Type of color scale to return
        
    Returns:
        Color scale as a list
    """
    if scale_type == "sequential":
        return VIRIDIS_DARK
    elif scale_type == "diverging":
        return DIVERGING_COLORS
    elif scale_type == "qualitative":
        return QUALITATIVE_COLORS
    else:
        # Default to sequential
        return VIRIDIS_DARK 