"""
Component for handling data uploads in the dashboard.

This module provides a file uploader widget that supports multiple file formats
and handles the processing of uploaded data files.
"""
from typing import Optional, Tuple, Union, Dict, Any, List, cast
import io
import pandas as pd
import streamlit as st


def data_uploader() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Creates a file uploader widget for CSV, Excel, and JSON data files.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        A tuple containing:
        - DataFrame with the uploaded data (None if no upload)
        - Filename of the uploaded file (None if no upload)
    
    Notes
    -----
    Supported file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    
    The component automatically detects the file type from the extension
    and uses the appropriate pandas reader function.
    
    Examples
    --------
    >>> df, filename = data_uploader()
    >>> if df is not None:
    ...     # Do something with the DataFrame
    ...     print(f"Loaded {filename} with {len(df)} rows")
    """
    st.subheader("Data Upload")
    
    with st.expander("Upload your data", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or JSON file",
            type=["csv", "xlsx", "xls", "json"],
            help="Upload your data to start analyzing"
        )
        
        if uploaded_file is not None:
            try:
                # Get file name and extension
                file_name = uploaded_file.name
                file_extension = file_name.split(".")[-1].lower()
                
                # Read different file types
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == "json":
                    df = pd.read_json(uploaded_file)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return None, None
                
                # Show success message with data dimensions
                st.success(f"Successfully loaded {file_name}: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Display preview of the data
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                return df, file_name
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None, None
        
        return None, None 