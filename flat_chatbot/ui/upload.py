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
    
    def _on_upload():
        """Callback for when files are uploaded"""
        if "_uploaded_files" in st.session_state:
            files = st.session_state["_uploaded_files"]
            any_new = controller.upload_files(files)
            if any_new:
                # No need for explicit rerun - callback will trigger it automatically
                pass
    
    st.file_uploader(
        "Upload CSV/Parquet", 
        type=["csv", "parquet"], 
        accept_multiple_files=True,
        key="_uploaded_files",
        on_change=_on_upload
    )

    col1, col2 = st.columns(2)
    
    def _on_clear():
        """Callback for clear all button"""
        controller.clear_all()
        st.success("Cleared")
    
    def _on_refresh():
        """Callback for refresh tables button"""
        controller.svc.initialize()
        st.success("Refreshed")
    
    with col1:
        st.button("Clear All Data", on_click=_on_clear)
    with col2:
        st.button("Refresh Tables", on_click=_on_refresh)

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
        
        def _on_tab_upload():
            """Callback for when files are uploaded in the tab"""
            if "_tab_uploaded_files" in st.session_state:
                process_uploaded_files(controller, st.session_state["_tab_uploaded_files"])
        
        # File uploader widget
        st.file_uploader(
            "Upload CSV files", 
            type=["csv"], 
            accept_multiple_files=True,
            key="_tab_uploaded_files",
            on_change=_on_tab_upload
        )


def process_uploaded_files(controller: AppController, uploaded_files: List[Any]) -> None:
    """
    Process uploaded files, display their information, and provide import functionality.
    
    Parameters
    ----------
    controller : AppController
        The application controller instance
    uploaded_files : List[Any]
        List of uploaded files from st.file_uploader
    """
    if not uploaded_files:
        return
        
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
    
    def _on_import():
        """Callback for when the import button is clicked"""
        with st.spinner("Importing data..."):
            success_count = 0
            for uploaded_file in uploaded_files:
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
                controller.svc.initialize()  # Initialize after import
            else:
                st.warning("No files were imported successfully")
    
    # Import button
    st.button("Import All to Database", on_click=_on_import, key="_import_button")
