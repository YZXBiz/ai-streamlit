"""Tests for external preprocessing assets."""

import sys
from pathlib import Path

import dagster as dg
import pandas as pd
import polars as pl
import pytest

# Add package directory to path if not already installed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clustering.dagster.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)
from tests.unit.preprocessing.conftest import MockReader, MockWriter


@pytest.fixture
def mock_external_data() -> pl.DataFrame:
    """Create sample external data.

    Returns:
        Sample external data for testing
    """
    return pl.DataFrame(
        {
            "store_id": ["S001", "S002", "S003"],
            "feature1": [10.0, 15.0, 12.0],
            "feature2": [100, 150, 120],
            "feature3": ["A", "B", "C"],
        }
    )


@pytest.fixture
def mock_external_data_list() -> list[pl.DataFrame]:
    """Create a list of sample external dataframes.

    Returns:
        List of external dataframes for testing
    """
    df1 = pl.DataFrame(
        {
            "store_id": ["S001", "S002", "S003"],
            "feature1": [10.0, 15.0, 12.0],
        }
    )
    df2 = pl.DataFrame(
        {
            "store_id": ["S001", "S002", "S003"],
            "feature2": [100, 150, 120],
        }
    )
    df3 = pl.DataFrame(
        {
            "store_id": ["S001", "S002", "S003"],
            "feature3": ["A", "B", "C"],
        }
    )
    return [df1, df2, df3]


def test_external_features_data_single(mock_external_data):
    """Test the external_features_data asset with a single input.

    Args:
        mock_external_data: Sample external data
    """
    # Create mock context with resources
    context = dg.build_op_context(
        resources={
            "input_external_sales_reader": MockReader(mock_external_data),
        }
    )

    # Run the asset
    result = external_features_data(context)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert "store_id" in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert "feature3" in result.columns
    assert len(result) == len(mock_external_data)


def test_external_features_data_multiple(mock_external_data_list, monkeypatch):
    """Test the external_features_data asset with multiple inputs.

    Args:
        mock_external_data_list: List of sample external dataframes
        monkeypatch: Pytest monkeypatch fixture
    """
    # Create mock readers
    mock_readers = [MockReader(df) for df in mock_external_data_list]

    # Expected merged result
    expected_merged = pl.DataFrame(
        {
            "store_id": ["S001", "S002", "S003"],
            "feature1": [10.0, 15.0, 12.0],
            "feature2": [100, 150, 120],
            "feature3": ["A", "B", "C"],
        }
    )

    # Mock the merge_dataframes function
    def mock_merge_dataframes(df_list):
        # Simple mock that combines all columns
        result = pd.DataFrame({"store_id": ["S001", "S002", "S003"]})
        for df in df_list:
            for col in df.columns:
                if col != "store_id":
                    result[col] = df[col].values
        return result

    # Apply the monkey patch
    from clustering.utils import helpers

    monkeypatch.setattr(helpers, "merge_dataframes", mock_merge_dataframes)

    # Create mock context with resources
    context = dg.build_op_context(
        resources={
            "input_external_sales_reader": mock_readers,
        }
    )

    # Run the asset
    result = external_features_data(context)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert "store_id" in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert "feature3" in result.columns
    assert len(result) == 3


def test_preprocessed_external_data(mock_external_data):
    """Test the preprocessed_external_data asset.

    Args:
        mock_external_data: Sample external data
    """
    # Create mock writer
    mock_writer = MockWriter()

    # Create mock context with resources
    context = dg.build_op_context(
        resources={
            "output_external_data_writer": mock_writer,
        }
    )

    # Run the asset
    result = preprocessed_external_data(context, mock_external_data)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert mock_writer.written_data is not None
    assert result is mock_external_data  # Should return the same dataframe


def test_preprocessed_external_data_pandas(mock_external_data):
    """Test the preprocessed_external_data asset with pandas writer.

    Args:
        mock_external_data: Sample external data
    """
    # Create mock writer that requires pandas
    mock_writer = MockWriter(requires_pandas=True)

    # Create mock context with resources
    context = dg.build_op_context(
        resources={
            "output_external_data_writer": mock_writer,
        }
    )

    # Run the asset
    result = preprocessed_external_data(context, mock_external_data)

    # Check the result
    assert isinstance(result, pl.DataFrame)
    assert mock_writer.written_data is not None
    assert isinstance(mock_writer.written_data, pd.DataFrame)  # Should be a pandas DataFrame
