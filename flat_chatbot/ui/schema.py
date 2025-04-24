"""
Schema UI component for the Data File/Table Chatbot application.

This module provides the UI components for viewing database schema information
in the application. It displays table structures, column definitions, and 
data previews for all loaded tables.
"""

from typing import Any, Dict
import streamlit as st
import pandas as pd

from flat_chatbot.controller import AppController


def render_schema_tab(controller: AppController, container: Any) -> None:
    """
    Renders the schema viewer tab in the Streamlit UI.
    
    This tab displays the current database schema, allowing users to view
    table structure information.
    """
    with container:
        st.header("Database Schema")
        
        # Get schema information
        if controller.svc is not None:
            schema_info = controller.get_schema()
            if schema_info and len(schema_info) > 0:
                for table_name, columns in schema_info.items():
                    st.subheader(f"Table: {table_name}")
                    
                    # Create a dataframe to display column information
                    if isinstance(columns, list):
                        # Create a DataFrame with just column names
                        df_schema = pd.DataFrame({"Column": columns})
                    else:
                        # Fallback for unexpected structure
                        df_schema = pd.DataFrame({"Column": ["Schema information has unexpected format"]})
                    
                    st.dataframe(df_schema)
                    
                    # Show sample data if available
                    if controller.svc is not None:
                        try:
                            sample_data = controller.svc.execute_query(f"SELECT * FROM {table_name} LIMIT 5")
                            if sample_data is not None and len(sample_data) > 0:
                                st.write("Sample data:")
                                st.dataframe(sample_data)
                        except Exception as e:
                            st.error(f"Error fetching sample data: {str(e)}")
            else:
                st.info("No tables found in the database. Please upload some data first.")
        else:
            st.warning("Database service not initialized. Please check your configuration.")
