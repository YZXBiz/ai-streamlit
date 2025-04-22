"""Data loading utilities for the dashboard."""

import pandas as pd
import streamlit as st
from io import BytesIO


def load_data_file(uploaded_file) -> pd.DataFrame:
    """Load data from an uploaded file.
    
    Args:
        uploaded_file: File object from st.file_uploader
        
    Returns:
        DataFrame containing the loaded data
    
    Raises:
        ValueError: If the file type is not supported
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_type == 'xlsx':
            return pd.read_excel(uploaded_file)
        elif file_type == 'parquet':
            return pd.read_parquet(BytesIO(uploaded_file.getvalue()))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        raise


def validate_cluster_data(df: pd.DataFrame) -> bool:
    """Validate that the dataframe contains the required columns for cluster analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ["STORE_NBR", "CLUSTER"]
    
    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Check if there are any clusters
    if df["CLUSTER"].nunique() < 2:
        st.warning("Data should have at least 2 clusters")
        return False
    
    return True 