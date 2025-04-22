"""Tests for the cluster visualization module."""
import pytest
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

from dashboard.components.cluster_viz import (
    create_cluster_scatter_plot,
    create_feature_importance_chart,
    create_cluster_summary_table
)


def test_create_cluster_scatter_plot():
    """Test that cluster scatter plot is created correctly."""
    # Create sample data
    df = pd.DataFrame({
        'CLUSTER_ID': [1, 1, 2, 2, 3],
        'FEATURE_1': [0.1, 0.2, 0.5, 0.6, 0.9],
        'FEATURE_2': [0.9, 0.8, 0.5, 0.4, 0.1],
        'STORE_NBR': ['S001', 'S002', 'S003', 'S004', 'S005']
    })
    
    # Call the function
    fig = create_cluster_scatter_plot(df, 'FEATURE_1', 'FEATURE_2')
    
    # Verify the result is a plotly figure
    assert isinstance(fig, go.Figure)
    
    # Verify the figure contains the correct number of traces (one per cluster + one for centroids)
    assert len(fig.data) == 4  # 3 clusters + centroids


def test_create_feature_importance_chart():
    """Test that feature importance chart is created correctly."""
    # Create sample data
    features = pd.DataFrame({
        'FEATURE': ['FEATURE_1', 'FEATURE_2', 'FEATURE_3'],
        'IMPORTANCE': [0.5, 0.3, 0.2]
    })
    
    # Call the function
    fig = create_feature_importance_chart(features)
    
    # Verify the result is a plotly figure
    assert isinstance(fig, go.Figure)
    
    # Verify the figure contains one trace
    assert len(fig.data) == 1
    
    # Verify the trace has the correct number of bars
    assert len(fig.data[0].y) == 3


@patch('dashboard.components.cluster_viz.st')
def test_create_cluster_summary_table(mock_st):
    """Test that cluster summary table is displayed correctly."""
    # Create a mock for streamlit AgGrid
    mock_aggrid = MagicMock()
    mock_st.AgGrid = mock_aggrid
    
    # Create sample data
    df = pd.DataFrame({
        'CLUSTER_ID': [1, 1, 2, 2, 3],
        'STORE_NBR': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'FEATURE_1': [0.1, 0.2, 0.5, 0.6, 0.9],
        'FEATURE_2': [0.9, 0.8, 0.5, 0.4, 0.1]
    })
    
    # Call the function
    create_cluster_summary_table(df)
    
    # Verify that AgGrid was called once
    mock_aggrid.assert_called_once()
    
    # Verify that the dataframe passed to AgGrid has the correct structure
    called_df = mock_aggrid.call_args[0][0]
    assert 'CLUSTER_ID' in called_df.columns
    assert 'Store Count' in called_df.columns
    assert 'FEATURE_1 (Mean)' in called_df.columns
    assert 'FEATURE_2 (Mean)' in called_df.columns 