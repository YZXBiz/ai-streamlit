"""Tests for the visualization module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from dashboard.components.visualization import (
    create_cluster_plot,
    create_feature_importance_plot,
    create_store_distribution_plot,
    create_correlation_heatmap,
    create_geographic_plot,
    create_performance_plot,
)


@patch("dashboard.components.visualization.plt")
@patch("dashboard.components.visualization.st")
def test_create_cluster_plot(mock_st, mock_plt):
    """Test cluster plot creation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "CLUSTER_ID": [1, 2, 1],
            "FEATURE_1": [0.1, 0.2, 0.3],
            "FEATURE_2": [0.4, 0.5, 0.6],
        }
    )

    # Create mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Create mock colormap
    mock_colormap = MagicMock()
    mock_plt.get_cmap.return_value = mock_colormap

    # Call function
    create_cluster_plot(df, x_feature="FEATURE_1", y_feature="FEATURE_2")

    # Verify plot creation
    mock_plt.subplots.assert_called_once()
    assert mock_ax.scatter.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_st.pyplot.called


@patch("dashboard.components.visualization.plt")
@patch("dashboard.components.visualization.st")
def test_create_feature_importance_plot(mock_st, mock_plt):
    """Test feature importance plot creation."""
    # Setup test data
    df = pd.DataFrame(
        {"FEATURE": ["Feature A", "Feature B", "Feature C"], "IMPORTANCE": [0.5, 0.3, 0.2]}
    )

    # Create mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Call function
    create_feature_importance_plot(df)

    # Verify plot creation
    mock_plt.subplots.assert_called_once()
    assert mock_ax.barh.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_st.pyplot.called


@patch("dashboard.components.visualization.plt")
@patch("dashboard.components.visualization.st")
def test_create_store_distribution_plot(mock_st, mock_plt):
    """Test store distribution plot creation."""
    # Setup test data
    df = pd.DataFrame(
        {"STORE_NBR": ["S001", "S002", "S003", "S004", "S005"], "CLUSTER_ID": [1, 2, 1, 3, 2]}
    )

    # Create mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Call function
    create_store_distribution_plot(df)

    # Verify plot creation
    mock_plt.subplots.assert_called_once()
    assert mock_ax.bar.called or mock_ax.pie.called
    assert mock_st.pyplot.called


@patch("dashboard.components.visualization.plt")
@patch("dashboard.components.visualization.st")
@patch("dashboard.components.visualization.sns")
def test_create_correlation_heatmap(mock_sns, mock_st, mock_plt):
    """Test correlation heatmap creation."""
    # Setup test data
    df = pd.DataFrame(
        {"FEATURE_1": [0.1, 0.2, 0.3], "FEATURE_2": [0.4, 0.5, 0.6], "FEATURE_3": [0.7, 0.8, 0.9]}
    )

    # Create mock figure
    mock_fig = MagicMock()
    mock_plt.figure.return_value = mock_fig

    # Call function
    create_correlation_heatmap(df)

    # Verify plot creation
    assert mock_sns.heatmap.called
    assert mock_st.pyplot.called


@patch("dashboard.components.visualization.px")
@patch("dashboard.components.visualization.st")
def test_create_geographic_plot(mock_st, mock_px):
    """Test geographic plot creation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "CLUSTER_ID": [1, 2, 1],
            "LATITUDE": [40.7128, 34.0522, 41.8781],
            "LONGITUDE": [-74.0060, -118.2437, -87.6298],
        }
    )

    # Create mock figure
    mock_fig = MagicMock()
    mock_px.scatter_mapbox.return_value = mock_fig

    # Call function
    create_geographic_plot(df)

    # Verify plot creation
    assert mock_px.scatter_mapbox.called
    assert mock_st.plotly_chart.called


@patch("dashboard.components.visualization.plt")
@patch("dashboard.components.visualization.st")
def test_create_performance_plot(mock_st, mock_plt):
    """Test performance plot creation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "CLUSTER_ID": [1, 2, 1],
            "SALES": [100000, 150000, 120000],
            "PROFIT": [10000, 15000, 12000],
        }
    )

    # Create mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Call function
    create_performance_plot(df, metric="SALES")

    # Verify plot creation
    mock_plt.subplots.assert_called_once()
    assert mock_ax.boxplot.called or mock_ax.violin.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_st.pyplot.called
