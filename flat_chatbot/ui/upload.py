"""
Upload UI component for the Data File/Table Chatbot application.

This module provides the UI components for file uploading and data management
in the application. It handles the sidebar interface for file uploads, table
listing, and data management operations.
"""

import streamlit as st
from typing import Any, List, Dict, Optional
import os
import pandas as pd

from flat_chatbot.controller import AppController
from flat_chatbot.services.duckdb_enhanced import EnhancedDuckDBService


def table_exists(svc: EnhancedDuckDBService, tbl: str) -> bool:
    """
    Check if a table exists in the DuckDB database.
    
    Parameters
    ----------
    svc : EnhancedDuckDBService
        The DuckDB service instance
    tbl : str
        Table name to check
        
    Returns
    -------
    bool
        True if the table exists, False otherwise
    """
    try:
        svc.execute_query(f"SELECT 1 FROM {tbl} LIMIT 0")
        return True
    except Exception:
        return False


def render_upload_sidebar(controller: AppController) -> None:
    """
    Render the data upload and management sidebar.
    
    Parameters
    ----------
    controller : AppController
        The application controller instance
    """
    st.markdown("<div class='section-header'>Data Management</div>", unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload CSV/Parquet", type=["csv", "parquet"], accept_multiple_files=True
    )
    controller.upload_files(files)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Data"):
            controller.clear_all()
            st.success("Cleared")
    with col2:
        if st.button("Refresh Tables"):
            controller.svc.initialize()
            st.success("Refreshed")

    tbls = controller.get_tables()
    if tbls:
        st.markdown("##### Available Tables")
        for i, t in enumerate(tbls, 1):
            st.markdown(f"{i}. **{t}**")
        mode = "advanced" if len(tbls) >= 2 else "simple"
        st.info(f"{'✨' if mode == 'advanced' else '🔍'} Using **{mode}** mode")


def render_upload_tab(controller: AppController, container: Any) -> None:
    """
    Renders the data upload tab in the Streamlit UI.
    
    This tab allows users to upload CSV files, view file details, 
    and import the data into the database.
    """
    with container:
        st.header("Upload Data")
        
        # File uploader widget
        uploaded_files = st.file_uploader(
            "Upload CSV files", 
            type=["csv"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Display information about uploaded files
            st.subheader("Uploaded Files")
            file_info = []
            
            for uploaded_file in uploaded_files:
                # Read and preview the data
                try:
                    df = pd.read_csv(uploaded_file)
                    row_count = len(df)
                    col_count = len(df.columns)
                    
                    file_info.append({
                        "filename": uploaded_file.name,
                        "rows": row_count,
                        "columns": col_count,
                        "preview": df.head(5)
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Display file summaries
            for info in file_info:
                with st.expander(f"{info['filename']} ({info['rows']} rows, {info['columns']} columns)"):
                    st.dataframe(info["preview"])
            
            # Import button
            if st.button("Import All to Database"):
                with st.spinner("Importing data..."):
                    success_count = 0
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            df = pd.read_csv(uploaded_file)
                            table_name = os.path.splitext(uploaded_file.name)[0]
                            
                            # Import to database through controller
                            if controller.svc is not None:
                                controller.import_dataframe(df, table_name)
                                success_count += 1
                                
                        except Exception as e:
                            st.error(f"Error importing {uploaded_file.name}: {str(e)}")
                    
                    if success_count > 0:
                        st.success(f"Successfully imported {success_count} file(s) to database")
                        st.rerun()  # Refresh the app to reflect the new data
                    else:
                        st.warning("No files were imported successfully")
