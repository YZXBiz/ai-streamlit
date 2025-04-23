#!/usr/bin/env python3
"""
Streamlit example for DuckDBService with LlamaIndex integration.

This example demonstrates using EnhancedDuckDBService with a Streamlit interface for:
1. Uploading data
2. Running SQL queries directly
3. Running natural language queries with LlamaIndex integration

Run with: streamlit run examples/streamlit_example.py
"""

import pandas as pd
import streamlit as st

from src.assortment_chatbot.services.duckdb_service import EnhancedDuckDBService

# Page configuration
st.set_page_config(page_title="DuckDB + LlamaIndex Demo", page_icon="🦙", layout="wide")
st.title("DuckDB + LlamaIndex Natural Language Query Demo")

# Initialize session state for storing our database service
if "db_service" not in st.session_state:
    st.session_state.db_service = EnhancedDuckDBService()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "tables" not in st.session_state:
    st.session_state.tables = []


# Define UI components in tabs
tab1, tab2, tab3 = st.tabs(["Data Loading", "SQL Query", "Natural Language Query"])

# Tab 1: Data Loading
with tab1:
    st.header("Load Data")

    # Option 1: Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Option 2: Use sample data
    use_sample = st.checkbox("Or use sample product data")

    table_name = st.text_input("Table name to create:", value="data")

    if st.button("Load Data"):
        if uploaded_file is not None:
            # Load data from uploaded file
            df = pd.read_csv(uploaded_file)
            success = st.session_state.db_service.load_dataframe(df, table_name)
            if success:
                st.success(f"Data loaded into table '{table_name}' successfully!")
                st.session_state.data_loaded = True
                # Get updated table list
                schema_info = st.session_state.db_service.get_schema_info()
                st.session_state.tables = schema_info["tables"]
            else:
                st.error("Failed to load data.")
        elif use_sample:
            # Create sample data
            data = {
                "product": ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"],
                "category": ["Electronics", "Electronics", "Electronics", "Audio", "Electronics"],
                "price": [1200, 800, 500, 150, 300],
                "stock": [25, 50, 35, 100, 20],
                "rating": [4.5, 4.2, 3.9, 4.7, 4.1],
            }
            df = pd.DataFrame(data)
            success = st.session_state.db_service.load_dataframe(df, table_name)
            if success:
                st.success(f"Sample data loaded into table '{table_name}' successfully!")
                st.session_state.data_loaded = True
                schema_info = st.session_state.db_service.get_schema_info()
                st.session_state.tables = schema_info["tables"]
                st.dataframe(df)
            else:
                st.error("Failed to load sample data.")
        else:
            st.warning("Please upload a file or use sample data.")

    # Display available tables
    if st.session_state.data_loaded:
        st.subheader("Available Tables")
        for table in st.session_state.tables:
            schema_info = st.session_state.db_service.get_schema_info()
            columns = schema_info["columns"].get(table, [])
            st.write(f"**Table: {table}**")
            st.write(f"Columns: {', '.join(columns)}")

# Tab 2: SQL Query
with tab2:
    st.header("Run SQL Query")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the 'Data Loading' tab.")
    else:
        sql_query = st.text_area(
            "Enter SQL Query:", value=f"SELECT * FROM {st.session_state.tables[0]} LIMIT 10"
        )

        if st.button("Run SQL Query"):
            result = st.session_state.db_service.execute_query(sql_query)
            if result is not None:
                st.success("Query executed successfully!")
                st.dataframe(result)
            else:
                st.error("Failed to execute query.")

# Tab 3: Natural Language Query
with tab3:
    st.header("Ask Questions in Natural Language")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the 'Data Loading' tab.")
    else:
        query = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What is the most expensive product?",
        )

        col1, col2 = st.columns(2)
        with col1:
            query_type = st.radio("Query Type", ["natural_language", "sql"], index=0)
        with col2:
            complexity = st.radio("Complexity", ["simple", "advanced"], index=0)

        if st.button("Run Query"):
            with st.spinner("Processing query..."):
                result = st.session_state.db_service.process_query(
                    query=query, query_type=query_type, complexity=complexity
                )

                if result["success"]:
                    st.success("Query processed successfully!")

                    # Display the answer
                    st.subheader("Answer")
                    st.write(result["data"])

                    # Display the generated SQL
                    st.subheader("Generated SQL")
                    st.code(result["sql_query"], language="sql")

                    # Display the raw data if available
                    if "raw_data" in result and result["raw_data"] is not None:
                        st.subheader("Raw Data")
                        st.dataframe(result["raw_data"])
                else:
                    st.error(f"Error processing query: {result.get('error', 'Unknown error')}")

# Display information about the implementation
with st.expander("About this Demo"):
    st.write("""
    This demo showcases using LlamaIndex with DuckDB for natural language to SQL translation.
    
    The implementation:
    1. Uses `NLSQLTableQueryEngine` for simple queries
    2. Uses `SQLTableRetrieverQueryEngine` with vector search for more complex queries
    3. Allows direct SQL execution as well
    
    This is powered by LlamaIndex's text-to-SQL capabilities, which use a language model to convert
    natural language questions into SQL queries based on the database schema.
    """)
