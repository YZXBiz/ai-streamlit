"""Tests for the DuckDB service with LlamaIndex integration."""

import pandas as pd
import pytest

from assortment_chatbot.services.duckdb_service import DuckDBService, EnhancedDuckDBService


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Product A", "Product B", "Product C", "Product D", "Product E"],
            "category": ["Electronics", "Clothing", "Electronics", "Food", "Toys"],
            "price": [100.0, 25.5, 350.0, 5.99, 15.0],
            "stock": [10, 50, 5, 100, 30],
        }
    )


@pytest.fixture
def sales_data():
    """Create sample sales data for testing."""
    return pd.DataFrame(
        {
            "sale_id": [101, 102, 103, 104, 105],
            "product_id": [1, 2, 3, 1, 5],
            "quantity": [2, 1, 1, 1, 3],
            "sale_date": ["2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-19"],
        }
    )


def test_basic_duckdb_service(sample_data):
    """Test basic DuckDB service functionality."""
    # Initialize service
    service = DuckDBService()

    # Load sample data
    success = service.load_dataframe(sample_data, "products")
    assert success is True

    # Execute a query
    result = service.execute_query("SELECT * FROM products WHERE category = 'Electronics'")
    assert len(result) == 2
    assert result["name"].tolist() == ["Product A", "Product C"]

    # Get schema info
    schema = service.get_schema_info()
    assert "products" in schema["tables"]
    assert "name" in schema["columns"]["products"]

    # Clear data
    success = service.clear_data()
    assert success is True

    # This should now raise an exception since the table no longer exists
    with pytest.raises(Exception):
        service.execute_query("SELECT * FROM products")


def test_enhanced_duckdb_service_initialization(sample_data):
    """Test enhanced DuckDB service initialization with LlamaIndex."""
    # Initialize service
    service = EnhancedDuckDBService()

    # Load sample data
    success = service.load_dataframe(sample_data, "products")
    assert success is True

    # Check if LlamaIndex components were initialized
    assert service.simple_query_engine is not None
    assert service.advanced_query_engine is not None
    assert service.table_schemas != {}
    assert "products" in service.table_schemas


def test_enhanced_duckdb_service_query(sample_data):
    """Test enhanced DuckDB service query capabilities."""
    # Initialize service
    service = EnhancedDuckDBService()

    # Load sample data
    success = service.load_dataframe(sample_data, "products")
    assert success is True

    # Test SQL query
    sql_result = service.process_query("SELECT * FROM products WHERE price > 100", query_type="sql")
    assert sql_result["success"] is True
    assert len(sql_result["data"]) == 1
    assert sql_result["data"]["name"].iloc[0] == "Product C"

    # Test natural language query
    # Note: This test requires LlamaIndex to be properly configured
    # We're just checking the structure of the response here
    nl_result = service.process_query("Show me electronics products", query_type="natural_language")
    assert "query_type" in nl_result
    assert nl_result["query_type"] == "natural_language"


def test_multi_table_queries(sample_data, sales_data):
    """Test enhanced DuckDB service with multiple tables."""
    # Initialize service
    service = EnhancedDuckDBService()

    # Load sample data
    service.load_dataframe(sample_data, "products")
    service.load_dataframe(sales_data, "sales")

    # Execute a SQL join query
    result = service.execute_query("""
        SELECT s.sale_id, p.name, s.quantity, p.price, s.quantity * p.price as total
        FROM sales s
        JOIN products p ON s.product_id = p.id
    """)

    assert len(result) == 5
    assert "total" in result.columns

    # Verify both tables are in the schema
    assert "products" in service.table_schemas
    assert "sales" in service.table_schemas


def test_clear_data(sample_data, sales_data):
    """Test clearing data from enhanced DuckDB service."""
    # Initialize service
    service = EnhancedDuckDBService()

    # Load sample data
    service.load_dataframe(sample_data, "products")
    service.load_dataframe(sales_data, "sales")

    # Clear data
    success = service.clear_data()
    assert success is True

    # Verify tables are gone from DuckDB
    assert service.tables == []

    # Verify LlamaIndex components are reset
    assert service.simple_query_engine is None
    assert service.advanced_query_engine is None
    assert service.table_index is None
    assert service.table_schemas == {}
