"""
Tests for the data loader utility.

This module contains tests for the data loading functions.
"""

import os
import tempfile
import pandas as pd
import pytest
import pandasai as pai

from frontend.utils.data_loader import load_file, create_sample_data


def test_create_sample_data():
    """Test that the sample data creation function works correctly."""
    # Call the function
    df = create_sample_data()
    
    # Check that it returns a PandasAI DataFrame
    assert isinstance(df, pai.DataFrame)
    
    # Check that it has the expected columns
    assert "country" in df._obj.columns
    assert "revenue" in df._obj.columns
    assert "employees" in df._obj.columns
    assert "year_founded" in df._obj.columns
    
    # Check that it has the expected number of rows
    assert len(df._obj) == 10


@pytest.mark.parametrize(
    "file_type,extension",
    [
        ("csv", "csv"),
        ("excel", "xlsx"),
        ("parquet", "parquet")
    ]
)
def test_load_file(file_type, extension, monkeypatch):
    """Test that the load_file function works correctly for different file types."""
    # Create a mock for the appropriate source class
    class MockSource:
        def __init__(self, file_path, name, description, **kwargs):
            self.file_path = file_path
            self.name = name
            self.description = description
            self.kwargs = kwargs
        
        def load(self):
            # Return a mock PandasAI DataFrame
            df = pd.DataFrame({"test": [1, 2, 3]})
            return pai.DataFrame(df, name=self.name, description=self.description)
    
    # Patch the appropriate source class
    if file_type == "csv":
        monkeypatch.setattr("frontend.utils.data_loader.CSVSource", MockSource)
    elif file_type == "excel":
        monkeypatch.setattr("frontend.utils.data_loader.ExcelSource", MockSource)
    elif file_type == "parquet":
        monkeypatch.setattr("frontend.utils.data_loader.ParquetSource", MockSource)
    
    # Call the function
    df = load_file(
        f"test.{extension}", 
        file_type, 
        "test_data", 
        "Test description",
        sheet_name="Sheet1" if file_type == "excel" else None
    )
    
    # Check that it returns a PandasAI DataFrame
    assert isinstance(df, pai.DataFrame)
    assert df._obj.shape == (3, 1)
    assert "test" in df._obj.columns
