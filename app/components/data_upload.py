"""
Data upload and connection component.

This module handles file uploads and Snowflake connections.
"""
import os
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO

from services.data_service import DataService

def render_data_upload():
    """Render the data upload and connection UI."""
    data_service = DataService()
    
    # Tab selection for data source
    source_tab = st.radio(
        "Select Data Source", 
        ["Upload File", "Snowflake Connection"],
        horizontal=True
    )
    
    if source_tab == "Upload File":
        render_file_upload(data_service)
    else:
        render_snowflake_connection(data_service)

def render_file_upload(data_service):
    """Render file upload interface."""
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Upload a data file to analyze"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            # Load the file based on its type
            if file_extension == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                data = pd.read_excel(uploaded_file)
            elif file_extension == "json":
                data = pd.read_json(uploaded_file)
            elif file_extension == "parquet":
                data = pd.read_parquet(BytesIO(uploaded_file.getvalue()))
            
            # Display dataset info
            st.success(f"Successfully loaded {uploaded_file.name}")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
            # Show sample of the data
            with st.expander("Preview Data"):
                st.dataframe(data.head())
            
            # Store data in session state
            data_service.store_data(data, source_type="file", source_name=uploaded_file.name)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return False
    
    return False

def render_snowflake_connection(data_service):
    """Render Snowflake connection interface."""
    with st.form("snowflake_connection"):
        st.subheader("Snowflake Connection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account = st.text_input("Account", help="Your Snowflake account identifier")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
        
        with col2:
            warehouse = st.text_input("Warehouse")
            database = st.text_input("Database")
            schema = st.text_input("Schema")
        
        query = st.text_area(
            "SQL Query", 
            height=150,
            help="Enter a SQL query to execute against Snowflake"
        )
        
        submit_button = st.form_submit_button("Connect & Run Query")
        
        if submit_button:
            # Check if all required fields are filled
            if not all([account, user, password, warehouse, database, schema, query]):
                st.error("Please fill all required fields")
                return False
            
            # Create credential dictionary
            credentials = {
                "account": account,
                "user": user,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema
            }
            
            try:
                with st.spinner("Connecting to Snowflake and executing query..."):
                    # Execute the query and get data
                    data = data_service.query_snowflake(credentials, query)
                    
                    if data is not None:
                        st.success("Successfully connected and executed query")
                        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
                        
                        # Show sample of the data
                        with st.expander("Preview Data"):
                            st.dataframe(data.head())
                        
                        # Store data in session state
                        data_service.store_data(
                            data, 
                            source_type="snowflake", 
                            source_name=f"{database}.{schema}"
                        )
                        
                        return True
                    else:
                        st.error("Query returned no data")
                        return False
            except Exception as e:
                st.error(f"Error connecting to Snowflake: {e}")
                return False
    
    return False 