"""Tests for the utility functions."""

import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, mock_open, MagicMock

from dashboard.utils.config import load_config, save_config, get_data_path

from dashboard.utils.helpers import (
    format_currency,
    format_percentage,
    calculate_growth,
    detect_outliers,
    create_store_lookup,
    export_results,
)


@patch("dashboard.utils.config.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_load_config(mock_file, mock_json_load):
    """Test config loading."""
    # Setup mock return value
    mock_config = {
        "data_path": "data/processed",
        "features": ["SALES", "PROFIT", "TRANSACTIONS"],
        "cluster_column": "CLUSTER_ID",
    }
    mock_json_load.return_value = mock_config

    # Call function
    config = load_config("mock_config.json")

    # Verify file opened correctly
    mock_file.assert_called_once_with("mock_config.json", "r")

    # Verify config content
    assert config == mock_config
    assert "data_path" in config
    assert "features" in config
    assert "cluster_column" in config


@patch("dashboard.utils.config.json.dump")
@patch("builtins.open", new_callable=mock_open)
def test_save_config(mock_file, mock_json_dump):
    """Test config saving."""
    # Setup test config
    test_config = {
        "data_path": "data/processed",
        "features": ["SALES", "PROFIT", "TRANSACTIONS"],
        "cluster_column": "CLUSTER_ID",
    }

    # Call function
    save_config(test_config, "mock_config.json")

    # Verify file opened correctly for writing
    mock_file.assert_called_once_with("mock_config.json", "w")

    # Verify json.dump called with correct arguments
    mock_json_dump.assert_called_once()
    args, _ = mock_json_dump.call_args
    assert args[0] == test_config


@patch("dashboard.utils.config.os.path.join")
@patch("dashboard.utils.config.load_config")
def test_get_data_path(mock_load_config, mock_path_join):
    """Test data path retrieval."""
    # Setup mock return values
    mock_load_config.return_value = {"data_path": "data/processed"}
    mock_path_join.return_value = "data/processed/stores.csv"

    # Call function
    result = get_data_path("stores.csv")

    # Verify correct path construction
    mock_load_config.assert_called_once()
    mock_path_join.assert_called_once_with("data/processed", "stores.csv")
    assert result == "data/processed/stores.csv"


def test_format_currency():
    """Test currency formatting."""
    # Test standard cases
    assert format_currency(1000) == "$1,000"
    assert format_currency(1234567.89) == "$1,234,568"  # Default rounds to nearest integer

    # Test with decimal places
    assert format_currency(1234.56, decimals=2) == "$1,234.56"

    # Test negative values
    assert format_currency(-500) == "-$500"

    # Test zero
    assert format_currency(0) == "$0"


def test_format_percentage():
    """Test percentage formatting."""
    # Test standard cases
    assert format_percentage(0.15) == "15.0%"
    assert format_percentage(0.5) == "50.0%"

    # Test with decimal places
    assert format_percentage(0.1234, decimals=2) == "12.34%"
    assert format_percentage(0.1234, decimals=0) == "12%"

    # Test values over 1
    assert format_percentage(1.5) == "150.0%"

    # Test negative values
    assert format_percentage(-0.25) == "-25.0%"


def test_calculate_growth():
    """Test growth calculation."""
    # Test standard growth
    assert calculate_growth(100, 150) == 0.5  # 50% growth

    # Test negative growth
    assert calculate_growth(200, 150) == -0.25  # 25% decline

    # Test zero previous value
    assert calculate_growth(0, 100) == float("inf")  # Infinite growth from zero

    # Test both zero
    assert calculate_growth(0, 0) == 0.0  # No growth when both are zero


def test_detect_outliers():
    """Test outlier detection."""
    # Setup test data
    data = pd.Series([10, 12, 15, 11, 14, 100, 13])

    # Call function
    outliers = detect_outliers(data)

    # Verify outlier detection
    assert len(outliers) == 1
    assert 100 in outliers.values


def test_create_store_lookup():
    """Test store lookup creation."""
    # Setup test data
    df = pd.DataFrame(
        {
            "STORE_NBR": ["S001", "S002", "S003"],
            "STORE_NAME": ["Store A", "Store B", "Store C"],
            "ADDRESS": ["123 Main St", "456 Elm St", "789 Oak St"],
            "CITY": ["New York", "Los Angeles", "Chicago"],
        }
    )

    # Call function
    lookup = create_store_lookup(df)

    # Verify lookup structure
    assert "S001" in lookup
    assert "S002" in lookup
    assert "S003" in lookup

    # Verify lookup content
    assert lookup["S001"]["name"] == "Store A"
    assert lookup["S002"]["address"] == "456 Elm St"
    assert lookup["S003"]["city"] == "Chicago"


@patch("dashboard.utils.helpers.pd.DataFrame.to_csv")
def test_export_results(mock_to_csv):
    """Test results export."""
    # Setup test data
    df = pd.DataFrame(
        {"STORE_NBR": ["S001", "S002", "S003"], "CLUSTER_ID": [1, 2, 1], "SALES": [100, 200, 300]}
    )

    # Call function
    export_results(df, "test_results.csv")

    # Verify export
    mock_to_csv.assert_called_once_with("test_results.csv", index=False)
