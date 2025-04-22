"""Tests for the UI components module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from dashboard.components.ui_components import (
    create_sidebar,
    create_header,
    create_filter_panel,
    apply_filters,
    create_metrics_panel,
    format_metric_display,
)


@patch("dashboard.components.ui_components.st")
def test_create_header(mock_st):
    """Test header creation."""
    # Call the function
    create_header("Test Title", "Test Subtitle")

    # Verify streamlit calls
    mock_st.title.assert_called_once_with("Test Title")
    mock_st.markdown.assert_called_with("Test Subtitle")


@patch("dashboard.components.ui_components.st")
def test_create_sidebar(mock_st):
    """Test sidebar creation."""
    # Create mock sidebar
    mock_sidebar = MagicMock()
    mock_st.sidebar = mock_sidebar

    # Set up test options
    options = {"Option 1": 1, "Option 2": 2}

    # Call the function
    result = create_sidebar("Test App", "v1.0", cluster_options=options, default_option="Option 1")

    # Verify streamlit calls
    mock_sidebar.title.assert_called_once_with("Test App")
    mock_sidebar.text.assert_called_with("v1.0")
    mock_sidebar.selectbox.assert_called_once()

    # Check function returns the expected selection
    assert isinstance(result, MagicMock)


@patch("dashboard.components.ui_components.st")
def test_create_filter_panel(mock_st):
    """Test filter panel creation."""
    # Setup test data
    df = pd.DataFrame({"FEATURE_1": [0.1, 0.2, 0.3], "FEATURE_2": ["A", "B", "A"]})

    numeric_cols = ["FEATURE_1"]
    categorical_cols = ["FEATURE_2"]

    # Call the function
    filters = create_filter_panel(df, numeric_cols, categorical_cols)

    # Verify streamlit calls
    assert mock_st.expander.called

    # Check filter structure
    assert "FEATURE_1" in filters
    assert "FEATURE_2" in filters
    assert isinstance(filters, dict)


def test_apply_filters():
    """Test filter application to dataframe."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003", "S004"],
            "CLUSTER_ID": [1, 2, 1, 2],
            "FEATURE_1": [0.1, 0.2, 0.3, 0.4],
            "FEATURE_2": ["A", "B", "A", "C"],
        }
    )

    # Create filters
    filters = {
        "FEATURE_1": (0.15, 0.35),  # Min and max for numeric
        "FEATURE_2": ["A"],  # Selected values for categorical
    }

    # Call the function
    filtered_df = apply_filters(df, filters)

    # Check filtering results
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["STORE_NBR"] == "S003"
    assert filtered_df.iloc[0]["FEATURE_1"] == 0.3
    assert filtered_df.iloc[0]["FEATURE_2"] == "A"


@patch("dashboard.components.ui_components.st")
def test_create_metrics_panel(mock_st):
    """Test metrics panel creation."""
    # Setup test data
    metrics = {"Metric 1": 42, "Metric 2": 3.14, "Metric 3": "87%"}

    # Call the function
    create_metrics_panel(metrics)

    # Verify streamlit calls
    assert mock_st.columns.called
    assert mock_st.metric.called


def test_format_metric_display():
    """Test metric formatting for display."""
    # Test integer formatting
    assert format_metric_display(42) == "42"

    # Test float formatting with default precision
    assert format_metric_display(3.14159) == "3.14"

    # Test float formatting with custom precision
    assert format_metric_display(3.14159, precision=3) == "3.142"

    # Test percentage formatting
    assert format_metric_display(0.7568, as_percentage=True) == "75.68%"

    # Test string passthrough
    assert format_metric_display("Custom Value") == "Custom Value"
