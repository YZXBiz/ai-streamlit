"""Data viewer component for displaying and exploring uploaded data."""

import pandas as pd
import streamlit as st


def fix_arrow_incompatible_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types that cause Arrow compatibility issues with Streamlit.
    
    Specifically converts pandas nullable integer types (Int64DType, etc.) to 
    standard numpy types to prevent PyArrow conversion errors.
    
    Args:
        df: Input DataFrame with potential Arrow-incompatible types
        
    Returns:
        DataFrame with Arrow-compatible data types
    """
    # Create a copy to avoid modifying the original
    df_fixed = df.copy()
    
    # Loop through each column and fix nullable integer types
    for col in df_fixed.columns:
        # Check if column type is a pandas nullable integer
        if hasattr(df_fixed[col].dtype, 'name'):
            if 'Int' in df_fixed[col].dtype.name:
                # Convert nullable integer to standard float (to preserve NaN values)
                df_fixed[col] = df_fixed[col].astype('float64')
        
        # Also fix object columns with mixed types that might cause issues
        if df_fixed[col].dtype == 'object':
            # Try to convert to string which is better handled by Arrow
            try:
                df_fixed[col] = df_fixed[col].astype('string')
            except (TypeError, ValueError):
                # If conversion fails, just leave as is
                pass
                
    return df_fixed


def data_viewer(df: pd.DataFrame) -> None:
    """
    Display and provide exploration options for uploaded data.
    
    This component allows users to:
    - View a sample of the data
    - See summary statistics
    - Filter and sort the data
    - Search for specific values
    
    Args:
        df: The pandas DataFrame containing the uploaded data
    """
    if df is None or df.empty:
        st.warning("No data to display. Please upload a file first.")
        return
    
    # Fix PyArrow incompatible types before displaying
    display_df = fix_arrow_incompatible_types(df)
    
    st.subheader("Data Preview")
    
    # Display data info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Rows:** {display_df.shape[0]} | **Columns:** {display_df.shape[1]}")
    
    with col2:
        st.info(f"**Memory usage:** {display_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Tabs for different views of the data
    tabs = st.tabs(["Data Sample", "Summary Statistics", "Column Info"])
    
    # Data Sample tab
    with tabs[0]:
        sample_size = min(10, len(display_df))
        st.dataframe(display_df.head(sample_size), use_container_width=True)
    
    # Summary Statistics tab
    with tabs[1]:
        # Only calculate numerical summaries to avoid errors
        numerical_cols = display_df.select_dtypes(include=['number']).columns.tolist()
        if numerical_cols:
            st.dataframe(display_df[numerical_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found for summary statistics.")
    
    # Column Info tab
    with tabs[2]:
        col_info = pd.DataFrame({
            'Column': display_df.columns,
            'Type': display_df.dtypes.values,
            'Non-Null Count': display_df.count().values,
            'Null Count': display_df.isna().sum().values,
            'Unique Values': [display_df[col].nunique() for col in display_df.columns]
        })
        st.dataframe(col_info, use_container_width=True) 