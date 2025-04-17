"""Dashboard utilities for the clustering application.

This module provides helper functions for the Streamlit dashboard.
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from clustering.dashboard.utils import get_color_scale


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from a file path.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Loaded DataFrame
    """
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    # Determine file type by extension
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.json':
        return pd.read_json(file_path)
    else:
        st.error(f"Unsupported file format: {file_path.suffix}")
        return pd.DataFrame()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_file_info(file_path: Path) -> dict[str, Any]:
    """Get file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    if not file_path.exists():
        return {"error": "File not found"}
    
    stats = file_path.stat()
    
    # Get basic file info
    info = {
        "name": file_path.name,
        "path": str(file_path),
        "size": format_file_size(stats.st_size),
        "size_bytes": stats.st_size,
        "created": datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
        "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        "extension": file_path.suffix.lower(),
    }
    
    return info


def show_dataframe_info(df: pd.DataFrame) -> None:
    """Display information about a DataFrame.
    
    Args:
        df: Input DataFrame
    """
    # Basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    
    # Show column types
    with st.expander("Column Data Types"):
        # Create a DataFrame with column info
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().sum().values / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(column_info)
    
    # Show summary statistics
    with st.expander("Summary Statistics"):
        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        else:
            st.info("No numeric columns found for statistics")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Plot missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
    """
    # Calculate missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        st.info("No missing values found in the dataset")
        return
    
    # Sort by missing count
    missing = missing.sort_values(ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={'x': 'Column', 'y': 'Missing Values'},
        title=f'Missing Values by Column (Total: {missing.sum():,})'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Column",
        yaxis_title="Count of Missing Values",
        xaxis={'categoryorder': 'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show percentage of missing values
    missing_percent = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_percent.values
    })
    
    st.dataframe(missing_df)


def display_code(code: str, language: str = "python") -> None:
    """Display code with syntax highlighting.
    
    Args:
        code: Code string to display
        language: Programming language for syntax highlighting
    """
    st.code(code, language=language)


def load_image(image_path: Path) -> Optional[Image.Image]:
    """Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded PIL Image or None if loading fails
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def download_dataframe(df: pd.DataFrame, file_name: str, file_format: str = "csv") -> None:
    """Provide a download button for a DataFrame.
    
    Args:
        df: DataFrame to download
        file_name: Name for the downloaded file (without extension)
        file_format: File format ('csv', 'excel', 'json', or 'parquet')
    """
    if file_format == "csv":
        data = df.to_csv(index=False)
        mime = "text/csv"
        extension = "csv"
    elif file_format == "excel":
        output = io.BytesIO()
        df.to_excel(output, index=False)
        data = output.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        extension = "xlsx"
    elif file_format == "json":
        data = df.to_json(orient="records", indent=2)
        mime = "application/json"
        extension = "json"
    elif file_format == "parquet":
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        data = output.getvalue()
        mime = "application/octet-stream"
        extension = "parquet"
    else:
        st.error(f"Unsupported file format: {file_format}")
        return
    
    st.download_button(
        label=f"Download as {file_format.upper()}",
        data=data,
        file_name=f"{file_name}.{extension}",
        mime=mime
    )


def create_figure_with_dropdown(df: pd.DataFrame, 
                              columns: list[str], 
                              color_col: Optional[str] = None,
                              title: str = "Interactive Plot with Dropdown") -> go.Figure:
    """Create a Plotly figure with dropdown to select features.
    
    Args:
        df: DataFrame with data
        columns: Columns to include in dropdown
        color_col: Column to use for color coding
        title: Figure title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add traces, one for each column
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines+markers',
                name=column,
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1),
                ),
                visible=(column == columns[0])  # Only first trace visible initially
            )
        )
    
    # Create buttons for dropdown
    buttons = []
    for i, column in enumerate(columns):
        visible = [False] * len(columns)
        visible[i] = True  # Only show the selected column
        buttons.append(
            dict(
                method="update",
                args=[{"visible": visible}, 
                      {"title": f"{title}: {column}"}],
                label=column
            )
        )
    
    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=f"{title}: {columns[0]}",
        height=600,
        hovermode="closest"
    )
    
    return fig


def create_sidebar_filters(df: pd.DataFrame, numeric_only: bool = False) -> dict[str, Any]:
    """Create filters in the sidebar for DataFrame columns.
    
    Args:
        df: DataFrame to create filters for
        numeric_only: Whether to only create filters for numeric columns
        
    Returns:
        Dictionary with filter values
    """
    filters = {}
    
    if numeric_only:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        columns = df.columns.tolist()
    
    if not columns:
        st.sidebar.warning("No suitable columns found for filtering")
        return filters
    
    st.sidebar.header("Filters")
    
    # Create filters based on data type
    for col in columns:
        # Skip if too many unique values (for categorical)
        if df[col].dtype == 'object' and df[col].nunique() > 30:
            continue
            
        # Create appropriate filter based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Skip if min and max are the same
            if min_val == max_val:
                continue
                
            # Create a slider
            if df[col].nunique() <= 10:
                # For few unique values, use selectbox
                options = sorted(df[col].unique())
                filters[col] = st.sidebar.multiselect(
                    f"Select {col}",
                    options=options,
                    default=options
                )
            else:
                # For continuous values, use a range slider
                filters[col] = st.sidebar.slider(
                    f"Filter by {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
        elif df[col].nunique() <= 30:
            # For categorical with reasonable number of categories
            options = sorted(df[col].unique())
            filters[col] = st.sidebar.multiselect(
                f"Select {col}",
                options=options,
                default=options
            )
    
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply filters to the DataFrame.
    
    Args:
        df: Input DataFrame
        filters: Dictionary of filters
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for col, filter_val in filters.items():
        if filter_val:  # Check if filter is not empty
            if isinstance(filter_val, tuple):
                # Range filter
                filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & 
                                         (filtered_df[col] <= filter_val[1])]
            elif isinstance(filter_val, list):
                # Multi-select filter
                if filter_val:  # Check if list is not empty
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
    
    return filtered_df


def cached_dataframe(func: Callable) -> Callable:
    """Decorator to cache DataFrame results.
    
    Args:
        func: Function that returns a DataFrame
        
    Returns:
        Wrapped function with caching
    """
    def wrapper(*args, **kwargs):
        # Create a cache key based on function arguments
        key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
        
        # Check if result is in session state
        if key not in st.session_state:
            # Run the function and store result
            result = func(*args, **kwargs)
            st.session_state[key] = result
        
        return st.session_state[key]
    
    return wrapper


def plot_correlation_matrix(df: pd.DataFrame, 
                          columns: Optional[list[str]] = None, 
                          method: str = "pearson") -> None:
    """Plot correlation matrix for selected columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to include in correlation matrix (defaults to all numeric)
        method: Correlation method ('pearson', 'kendall', 'spearman')
    """
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(columns) < 2:
        st.warning("Need at least 2 numeric columns to calculate correlations")
        return
    
    # Calculate correlation matrix
    corr = df[columns].corr(method=method)
    
    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=get_color_scale('diverging'),
        labels=dict(color="Correlation"),
        title=f"{method.capitalize()} Correlation Matrix"
    )
    
    # Update layout
    fig.update_layout(
        height=700 if len(columns) > 10 else 500,
        width=700 if len(columns) > 10 else 500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Allow downloading the correlation matrix
    with st.expander("View correlation values"):
        st.dataframe(corr.style.highlight_max(axis=None))
        
        # Download option
        csv = corr.to_csv(index=True)
        st.download_button(
            label="Download Correlation Matrix",
            data=csv,
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )


def create_time_series_plot(df: pd.DataFrame, 
                          date_col: str,
                          value_col: str,
                          group_col: Optional[str] = None,
                          title: str = "Time Series Analysis") -> None:
    """Create an interactive time series plot.
    
    Args:
        df: Input DataFrame
        date_col: Column with dates
        value_col: Column with values to plot
        group_col: Optional column to group by (for multiple lines)
        title: Plot title
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.error(f"Could not convert {date_col} to datetime format")
            return
    
    # Create the time series plot
    if group_col:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            color=group_col,
            title=title,
            labels={
                date_col: "Date",
                value_col: "Value",
                group_col: "Group"
            }
        )
    else:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=title,
            labels={
                date_col: "Date",
                value_col: "Value"
            }
        )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add time aggregation option
    with st.expander("Time Aggregation"):
        # Let user select aggregation method and period
        agg_method = st.selectbox(
            "Aggregation Method",
            options=["mean", "sum", "min", "max", "count"],
            index=0
        )
        
        agg_period = st.selectbox(
            "Aggregation Period",
            options=["Day", "Week", "Month", "Quarter", "Year"],
            index=2
        )
        
        # Create aggregated view
        df_copy = df.copy()
        
        # Set the date as index for easier resampling
        df_copy = df_copy.set_index(date_col)
        
        # Map aggregation period to pandas resample rule
        period_map = {
            "Day": "D",
            "Week": "W",
            "Month": "M",
            "Quarter": "Q",
            "Year": "Y"
        }
        
        # Perform resampling based on group column
        if group_col:
            # Group by the group column first
            agg_df = df_copy.groupby(group_col).resample(
                period_map[agg_period]
            )[value_col].agg(agg_method).reset_index()
            
            # Create new plot
            agg_fig = px.line(
                agg_df,
                x=date_col,
                y=value_col,
                color=group_col,
                title=f"{title} ({agg_method} by {agg_period})",
                labels={
                    date_col: "Date",
                    value_col: f"{agg_method.capitalize()} of {value_col}",
                    group_col: "Group"
                }
            )
            
        else:
            # Simple resampling without groups
            agg_df = df_copy.resample(period_map[agg_period])[value_col].agg(agg_method)
            agg_df = agg_df.reset_index()
            
            # Create new plot
            agg_fig = px.line(
                agg_df,
                x=date_col,
                y=value_col,
                title=f"{title} ({agg_method} by {agg_period})",
                labels={
                    date_col: "Date",
                    value_col: f"{agg_method.capitalize()} of {value_col}"
                }
            )
        
        st.plotly_chart(agg_fig, use_container_width=True)


def export_dashboard_state(state_dict: dict[str, Any], file_name: str = "dashboard_state") -> None:
    """Export dashboard state to a file.
    
    Args:
        state_dict: Dictionary with state to export
        file_name: Base name for the exported file
    """
    # Convert to JSON
    state_json = json.dumps(state_dict, indent=2, default=str)
    
    # Provide download button
    st.download_button(
        label="Export Dashboard State",
        data=state_json,
        file_name=f"{file_name}.json",
        mime="application/json"
    )


def import_dashboard_state(default_state: dict[str, Any]) -> dict[str, Any]:
    """Import dashboard state from a file.
    
    Args:
        default_state: Default state to use if import fails
        
    Returns:
        Imported state dictionary or default state
    """
    uploaded_file = st.file_uploader("Import Dashboard State", type=["json"])
    
    if uploaded_file is not None:
        try:
            state_dict = json.load(uploaded_file)
            st.success("Dashboard state imported successfully")
            return state_dict
        except Exception as e:
            st.error(f"Error importing dashboard state: {e}")
    
    return default_state 