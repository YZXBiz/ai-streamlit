#!/usr/bin/env python3
"""
Example usage of DuckDBService with LlamaIndex integration.

This example demonstrates:
1. Loading sample data into DuckDB
2. Running SQL queries directly
3. Running natural language queries with LlamaIndex integration
"""

import pandas as pd

from src.assortment_chatbot.services.duckdb_service import EnhancedDuckDBService


def main():
    """Example usage of EnhancedDuckDBService."""
    # Create sample data
    data = {
        "product": ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"],
        "category": ["Electronics", "Electronics", "Electronics", "Audio", "Electronics"],
        "price": [1200, 800, 500, 150, 300],
        "stock": [25, 50, 35, 100, 20],
        "rating": [4.5, 4.2, 3.9, 4.7, 4.1],
    }
    df = pd.DataFrame(data)
    print("Sample data created:")
    print(df.head())
    print("-" * 50)

    # Initialize the enhanced DuckDB service (in-memory database)
    db_service = EnhancedDuckDBService()

    # Load data into the service
    success = db_service.load_dataframe(df, "products")
    if success:
        print("Data loaded successfully.")
    else:
        print("Failed to load data.")
        return

    # Get schema information
    schema_info = db_service.get_schema_info()
    print("\nSchema information:")
    print(f"Tables: {schema_info['tables']}")
    print(f"Columns in products: {schema_info['columns']['products']}")
    print("-" * 50)

    # Direct SQL query
    print("\nRunning direct SQL query:")
    sql_query = "SELECT * FROM products WHERE price > 500 ORDER BY price DESC"
    print(f"SQL: {sql_query}")
    results = db_service.execute_query(sql_query)
    print(results)
    print("-" * 50)

    # Natural language query using LlamaIndex integration
    print("\nRunning natural language queries:")

    # Simple query
    nl_query = "What is the most expensive product?"
    print(f"\nQuestion: {nl_query}")
    result = db_service.process_query(
        query=nl_query, query_type="natural_language", complexity="simple"
    )

    if result["success"]:
        print(f"Answer: {result['data']}")
        print(f"Generated SQL: {result['sql_query']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # More complex query
    nl_query = "What is the average price of electronics products with rating above 4?"
    print(f"\nQuestion: {nl_query}")
    result = db_service.process_query(
        query=nl_query, query_type="natural_language", complexity="advanced"
    )

    if result["success"]:
        print(f"Answer: {result['data']}")
        print(f"Generated SQL: {result['sql_query']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
