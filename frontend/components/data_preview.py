"""
Data preview component for the PandasAI Streamlit application.

This module provides the data preview component for viewing loaded datasets.
"""

import streamlit as st
import pandas as pd


def render_data_preview():
    """
    Render the data preview component.
    
    This function displays:
    - A dropdown to select a dataset
    - A preview of the selected dataset
    - Schema information and basic statistics
    """
    # Get available dataframes
    available_dfs = st.session_state.loaded_dataframes
    
    if not available_dfs:
        st.info("No datasets available for preview. Please upload data first.")
        return
    
    # Data preview
    selected_df = st.selectbox("Select dataset to preview:", available_dfs)
    
    if selected_df:
        df_obj = st.session_state.analyzer.dataframe_manager.get_dataframe(selected_df)
        if df_obj is not None:
            # Get the pandas DataFrame from the PandasAI DataFrame
            # In PandasAI v3, the underlying pandas DataFrame is accessed via ._obj
            pandas_df = df_obj._obj if hasattr(df_obj, "_obj") else df_obj
            
            # Display dataframe info
            st.markdown(f"### Preview of {selected_df}")
            st.dataframe(pandas_df.head(10))
            
            # Display schema info
            with st.expander("Dataset Schema"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Columns")
                    st.write(pandas_df.columns.tolist())
                
                with col2:
                    st.markdown("#### Data Types")
                    st.write(pandas_df.dtypes.to_dict())
            
            # Display basic statistics
            with st.expander("Basic Statistics"):
                st.write(pandas_df.describe())
            
            # Display sample queries
            with st.expander("Sample Queries"):
                st.markdown("### Try asking:")
                
                # Generate sample queries based on dataframe content
                columns = pandas_df.columns.tolist()
                numeric_cols = pandas_df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = pandas_df.select_dtypes(include=['object', 'category']).columns.tolist()
                date_cols = pandas_df.select_dtypes(include=['datetime']).columns.tolist()
                
                if numeric_cols:
                    st.markdown(f"- What is the average of {numeric_cols[0]}?")
                    if len(numeric_cols) > 1:
                        st.markdown(f"- Show me the correlation between {numeric_cols[0]} and {numeric_cols[1]}")
                    st.markdown(f"- What is the distribution of {numeric_cols[0]}?")
                
                if categorical_cols:
                    st.markdown(f"- Show me the count of each {categorical_cols[0]}")
                    if numeric_cols:
                        st.markdown(f"- What is the average {numeric_cols[0]} by {categorical_cols[0]}?")
                
                if date_cols:
                    st.markdown(f"- Show me the trend of {numeric_cols[0] if numeric_cols else columns[0]} over time")
                
                st.markdown("- Summarize this dataset")
                st.markdown("- What insights can you find in this data?")
