"""
Tests for data service functionality.
"""
import pandas as pd
import pytest
from pathlib import Path

# Import the service to test
try:
    from app.services.data_service import DataService
except ImportError:
    # For development, we might need to adjust the import path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.services.data_service import DataService

@pytest.fixture
def sample_data():
    """Fixture for sample data DataFrame."""
    return pd.DataFrame({
        'product_id': [1, 2, 3],
        'product_name': ['Laptop', 'Mouse', 'Keyboard'],
        'category': ['Electronics', 'Electronics', 'Electronics'],
        'price': [1200.0, 25.99, 89.99],
        'inventory': [45, 120, 40],
        'rating': [4.7, 4.3, 4.4],
    })

@pytest.fixture
def data_service():
    """Fixture for DataService instance."""
    return DataService()

def test_store_data(data_service, sample_data):
    """Test storing data in the service."""
    # Store the data
    result = data_service.store_data(sample_data, "test", "sample_data")
    
    # Check that storage was successful
    assert result is True
    
    # Check that DB connection was created
    assert data_service.db_conn is not None

def test_execute_query(data_service, sample_data):
    """Test executing a query against the data."""
    # Store the data first
    data_service.store_data(sample_data, "test", "sample_data")
    
    # Execute a simple query
    result, error = data_service.execute_query("SELECT * FROM sample_data")
    
    # Check there's no error
    assert error is None
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that result has the same shape as input
    assert result.shape == sample_data.shape

def test_get_schema_info(data_service, sample_data, monkeypatch):
    """Test getting schema information."""
    # Mock the session state with the necessary data
    class MockSessionState:
        data = sample_data
        data_metadata = {
            "table_name": "sample_data",
            "source_type": "test",
            "source_name": "sample_data",
            "row_count": len(sample_data),
            "column_count": len(sample_data.columns),
            "columns": list(sample_data.columns)
        }
    
    # Mock the session_state in streamlit
    monkeypatch.setattr("streamlit.session_state", MockSessionState())
    
    # Get schema info
    schema_info = data_service.get_schema_info()
    
    # Check schema info is not empty
    assert schema_info
    
    # Check table name
    assert schema_info["table_name"] == "sample_data"
    
    # Check all columns are present
    assert set(schema_info["columns"].keys()) == set(sample_data.columns)
    
    # Check numeric columns have correct type category
    assert schema_info["columns"]["price"]["type_category"] == "numeric"
    assert schema_info["columns"]["inventory"]["type_category"] == "numeric"
    
    # Check string columns have correct type category
    assert schema_info["columns"]["product_name"]["type_category"] == "string" 