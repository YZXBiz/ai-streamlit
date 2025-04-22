"""Tests for the data loader module."""

import io
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

from dashboard.data.loader import load_data_file, validate_cluster_data
from dashboard.components.data_loader import (
    load_cluster_data,
    load_feature_importance,
    validate_data_schema,
    normalize_features
)


def test_validate_cluster_data_valid(sample_cluster_data):
    """Test that a valid DataFrame passes validation."""
    assert validate_cluster_data(sample_cluster_data) is True


def test_validate_cluster_data_missing_columns():
    """Test that a DataFrame missing required columns fails validation."""
    df = pd.DataFrame({
        "STORE_NBR": ["123", "456"],
        "SALES": [100000, 200000],
    })
    with patch("streamlit.error") as mock_error:
        assert validate_cluster_data(df) is False
        mock_error.assert_called_once()


def test_validate_cluster_data_too_few_clusters():
    """Test that a DataFrame with too few clusters fails validation."""
    df = pd.DataFrame({
        "STORE_NBR": ["123", "456", "789"],
        "CLUSTER": [1, 1, 1],  # Only one cluster
    })
    with patch("streamlit.warning") as mock_warning:
        assert validate_cluster_data(df) is False
        mock_warning.assert_called_once()


class MockUploadedFile:
    """Mock for a file uploaded through st.file_uploader."""
    
    def __init__(self, name, content):
        self.name = name
        self.content = content
    
    def getvalue(self):
        return self.content
    
    def read(self):
        return self.content


@pytest.mark.parametrize(
    "file_name,file_content,expected_shape",
    [
        (
            "test.csv",
            b"STORE_NBR,CLUSTER\n123,0\n456,1",
            (2, 2)
        ),
    ]
)
def test_load_data_file_csv(file_name, file_content, expected_shape):
    """Test loading CSV data."""
    uploaded_file = MockUploadedFile(file_name, file_content)
    
    with patch("pandas.read_csv", return_value=pd.DataFrame({
        "STORE_NBR": ["123", "456"],
        "CLUSTER": [0, 1]
    })) as mock_read_csv:
        df = load_data_file(uploaded_file)
        mock_read_csv.assert_called_once()
        assert df.shape == expected_shape 


@patch('dashboard.components.data_loader.pd.read_csv')
def test_load_cluster_data(mock_read_csv):
    """Test loading cluster data from CSV."""
    # Setup mock return value
    mock_df = pd.DataFrame({
        'STORE_NBR': ['S001', 'S002', 'S003'],
        'CLUSTER_ID': [1, 2, 1],
        'FEATURE_1': [0.5, 0.7, 0.4],
        'FEATURE_2': [0.3, 0.2, 0.8]
    })
    mock_read_csv.return_value = mock_df
    
    # Call the function
    result = load_cluster_data('fake_path.csv')
    
    # Verify the result
    assert mock_read_csv.called
    pd.testing.assert_frame_equal(result, mock_df)
    
    # Verify STORE_NBR is treated as string
    assert result['STORE_NBR'].dtype == 'object'


@patch('dashboard.components.data_loader.pd.read_csv')
def test_load_feature_importance(mock_read_csv):
    """Test loading feature importance data from CSV."""
    # Setup mock return value
    mock_df = pd.DataFrame({
        'FEATURE': ['FEATURE_1', 'FEATURE_2', 'FEATURE_3'],
        'IMPORTANCE': [0.6, 0.3, 0.1]
    })
    mock_read_csv.return_value = mock_df
    
    # Call the function
    result = load_feature_importance('fake_path.csv')
    
    # Verify the result
    assert mock_read_csv.called
    pd.testing.assert_frame_equal(result, mock_df)


def test_validate_data_schema_valid():
    """Test schema validation with valid data."""
    # Create test data with valid schema
    df = pd.DataFrame({
        'STORE_NBR': ['S001', 'S002', 'S003'],
        'CLUSTER_ID': [1, 2, 1],
        'FEATURE_1': [0.5, 0.7, 0.4],
        'FEATURE_2': [0.3, 0.2, 0.8]
    })
    
    required_columns = ['STORE_NBR', 'CLUSTER_ID']
    
    # Should not raise exception
    validate_data_schema(df, required_columns)


def test_validate_data_schema_invalid():
    """Test schema validation with invalid data."""
    # Create test data with invalid schema (missing CLUSTER_ID)
    df = pd.DataFrame({
        'STORE_NBR': ['S001', 'S002', 'S003'],
        'FEATURE_1': [0.5, 0.7, 0.4],
        'FEATURE_2': [0.3, 0.2, 0.8]
    })
    
    required_columns = ['STORE_NBR', 'CLUSTER_ID']
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        validate_data_schema(df, required_columns)


def test_normalize_features():
    """Test feature normalization."""
    # Create test data
    df = pd.DataFrame({
        'STORE_NBR': ['S001', 'S002', 'S003'],
        'CLUSTER_ID': [1, 2, 1],
        'FEATURE_1': [10, 20, 30],
        'FEATURE_2': [100, 200, 300]
    })
    
    feature_columns = ['FEATURE_1', 'FEATURE_2']
    
    # Call the function
    result = normalize_features(df, feature_columns)
    
    # Check non-feature columns are unchanged
    assert 'STORE_NBR' in result.columns
    assert 'CLUSTER_ID' in result.columns
    
    # Check feature columns are normalized
    assert result['FEATURE_1'].min() == 0.0
    assert result['FEATURE_1'].max() == 1.0
    assert result['FEATURE_2'].min() == 0.0
    assert result['FEATURE_2'].max() == 1.0 