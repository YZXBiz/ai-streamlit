"""Tests for the data processing module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from dashboard.components.data_processing import (
    load_data,
    preprocess_data,
    compute_statistics,
    filter_by_cluster,
    calculate_cluster_metrics,
    normalize_features,
)


@patch("dashboard.components.data_processing.pd.read_csv")
def test_load_data(mock_read_csv):
    """Test data loading."""
    # Setup mock return value
    mock_df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "FEATURE_1": [10, 20, 30],
            "FEATURE_2": [40, 50, 60],
        }
    )
    mock_read_csv.return_value = mock_df

    # Call function
    result = load_data("mock_path.csv")

    # Verify correct function call and result
    mock_read_csv.assert_called_once_with("mock_path.csv")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_df)


def test_preprocess_data():
    """Test data preprocessing."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "FEATURE_1": [10, np.nan, 30],
            "FEATURE_2": [40, 50, np.nan],
            "EXTRA_COL": ["A", "B", "C"],
        }
    )

    # Define required columns
    required_columns = ["STORE_NBR", "FEATURE_1", "FEATURE_2"]

    # Call function
    result = preprocess_data(df, required_columns)

    # Verify preprocessing steps
    assert set(result.columns) == set(required_columns)
    assert not result.isna().any().any()
    assert result.shape[0] == 1  # Only one row should remain after removing NaNs


def test_compute_statistics():
    """Test statistics computation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "CLUSTER_ID": [1, 2, 1],
            "SALES": [100, 200, 300],
            "PROFIT": [10, 20, 30],
        }
    )

    # Call function
    result = compute_statistics(df)

    # Verify statistics computation
    assert "MEAN_SALES" in result
    assert "MEAN_PROFIT" in result
    assert "CLUSTER_COUNT" in result
    assert "STORE_COUNT" in result
    assert result["STORE_COUNT"] == 3
    assert result["CLUSTER_COUNT"] == 2


def test_filter_by_cluster():
    """Test cluster filtering."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003", "S004"],
            "CLUSTER_ID": [1, 2, 1, 3],
            "FEATURE": [10, 20, 30, 40],
        }
    )

    # Call function
    result = filter_by_cluster(df, cluster_id=1)

    # Verify filtering
    assert result.shape[0] == 2
    assert all(result["CLUSTER_ID"] == 1)
    assert set(result["STORE_NBR"]) == {"S001", "S003"}


def test_calculate_cluster_metrics():
    """Test cluster metrics calculation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003", "S004"],
            "CLUSTER_ID": [1, 2, 1, 2],
            "SALES": [100, 200, 300, 400],
            "PROFIT": [10, 20, 30, 40],
            "TRANSACTIONS": [1000, 2000, 3000, 4000],
        }
    )

    # Call function
    result = calculate_cluster_metrics(df)

    # Verify metrics calculation
    assert result.shape[0] == 2  # Two clusters
    assert set(result["CLUSTER_ID"]) == {1, 2}
    assert "STORE_COUNT" in result.columns
    assert "AVG_SALES" in result.columns
    assert "AVG_PROFIT" in result.columns
    assert "AVG_TRANSACTIONS" in result.columns

    # Verify specific metrics for cluster 1
    cluster_1 = result[result["CLUSTER_ID"] == 1].iloc[0]
    assert cluster_1["STORE_COUNT"] == 2
    assert cluster_1["AVG_SALES"] == 200  # (100 + 300) / 2
    assert cluster_1["AVG_PROFIT"] == 20  # (10 + 30) / 2


def test_normalize_features():
    """Test feature normalization."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "FEATURE_1": [10, 20, 30],
            "FEATURE_2": [100, 200, 300],
        }
    )

    # Define features to normalize
    features = ["FEATURE_1", "FEATURE_2"]

    # Call function
    result = normalize_features(df, features)

    # Verify normalization
    assert "STORE_NBR" in result.columns
    assert "FEATURE_1" in result.columns
    assert "FEATURE_2" in result.columns

    # Check if values are normalized (between 0 and 1)
    for feature in features:
        assert result[feature].min() >= 0
        assert result[feature].max() <= 1

    # Check relative positions are maintained after normalization
    assert result.loc[0, "FEATURE_1"] < result.loc[1, "FEATURE_1"] < result.loc[2, "FEATURE_1"]
    assert result.loc[0, "FEATURE_2"] < result.loc[1, "FEATURE_2"] < result.loc[2, "FEATURE_2"]
