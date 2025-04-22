"""
Data visualization component using PyWalker.

This module handles interactive data visualization.
"""
import streamlit as st
import pandas as pd
from pywalker import api as pwk

def render_visualization(data: pd.DataFrame):
    """
    Render the visualization interface using PyWalker.
    
    Args:
        data: The DataFrame to visualize
    """
    if data is None or data.empty:
        st.warning("No data available for visualization.")
        return
    
    st.header("Data Visualization")
    
    # Data info section
    with st.expander("Dataset Information", expanded=False):
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        # Display column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isna().sum(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info)
    
    # PyWalker visualization
    st.subheader("Interactive Visualization")
    
    # Generate PyWalker visualization
    try:
        # Optional: Create simple sample if data is very large
        if data.shape[0] > 10000:
            sampled_data = data.sample(10000, random_state=42)
            st.info(f"Data is large. Visualizing a random sample of 10,000 rows from {data.shape[0]} total rows.")
        else:
            sampled_data = data
        
        # Initialize PyWalker with the DataFrame
        vis = pwk.to_html(
            sampled_data,
            return_html=True,
            dark="auto",
            use_kernel=True
        )
        
        # Display in Streamlit
        st.components.v1.html(vis, height=600, scrolling=True)
        
        # Save visualization state
        update_visualization_state("pywalker", {}, [])
        
    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        
        # Fallback simple visualizations
        st.subheader("Basic Data Preview")
        st.dataframe(data.head(100))

def update_visualization_state(chart_type, filters, aggregations):
    """
    Update the visualization state in session state.
    
    Args:
        chart_type: The type of chart being displayed
        filters: Dictionary of applied filters
        aggregations: List of applied aggregations
    """
    st.session_state.visualization_state = {
        "last_chart_type": chart_type,
        "filters": filters,
        "aggregations": aggregations
    } 