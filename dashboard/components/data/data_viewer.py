"""
Component for viewing and exploring data in the dashboard.

This module provides interactive components for data exploration, visualization,
and analysis through a user-friendly interface.
"""

from typing import Optional, List, Dict, Any, Union, Tuple, cast
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objects import Figure


def data_viewer(df: pd.DataFrame | None = None) -> None:
    """
    Creates a data viewer component with exploration and visualization options.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        The DataFrame to visualize and explore. If None, shows a message
        prompting the user to upload data.

    Returns
    -------
    None
        This function modifies the Streamlit UI but doesn't return any values.

    Notes
    -----
    This component provides:
    - Data summary statistics
    - Column information (types, null counts, unique values)
    - Interactive visualizations (histograms, box plots, scatter plots, bar charts)

    The visualization options are context-aware and will adapt based on the
    data types present in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'numeric': [1, 2, 3], 'category': ['A', 'B', 'C']})
    >>> data_viewer(df)
    """
    st.subheader("Data Explorer")

    if df is None:
        st.info("Please upload data to use the data explorer.")
        return

    # Data summary tab
    with st.expander("Data Summary", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Missing Values", df.isna().sum().sum())

        with col2:
            st.metric("Columns", df.shape[1])
            st.metric("Duplicate Rows", df.duplicated().sum())

        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame(
            {
                "Type": [str(dtype) for dtype in df.dtypes],
                "Non-Null Count": df.count(),
                "Null Count": df.isna().sum(),
                "Unique Values": [df[col].nunique() for col in df.columns],
            }
        )
        st.dataframe(col_info, use_container_width=True)

    # Quick visualizations
    with st.expander("Quick Visualizations", expanded=True):
        # Column selection
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found for visualization.")
            return

        # Visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            options=["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"],
            index=0,
        )

        if viz_type == "Histogram":
            _create_histogram(df, numeric_cols)

        elif viz_type == "Box Plot":
            _create_box_plot(df, numeric_cols, categorical_cols)

        elif viz_type == "Scatter Plot":
            _create_scatter_plot(df, numeric_cols, categorical_cols)

        elif viz_type == "Bar Chart":
            _create_bar_chart(df, categorical_cols)


def _create_histogram(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """
    Create and display a histogram visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_cols : List[str]
        List of numeric column names available for visualization

    Returns
    -------
    None
        This function displays the visualization in the Streamlit UI
    """
    col = st.selectbox("Select column for histogram", options=numeric_cols)
    fig = px.histogram(df, x=col, marginal="box")
    st.plotly_chart(fig, use_container_width=True)


def _create_box_plot(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> None:
    """
    Create and display a box plot visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_cols : List[str]
        List of numeric column names available for visualization
    categorical_cols : List[str]
        List of categorical column names available for grouping

    Returns
    -------
    None
        This function displays the visualization in the Streamlit UI
    """
    col = st.selectbox("Select column for box plot", options=numeric_cols)
    group_by = None
    if categorical_cols:
        group_by = st.selectbox("Group by (optional)", options=["None"] + categorical_cols)

    if group_by and group_by != "None":
        fig = px.box(df, x=group_by, y=col)
    else:
        fig = px.box(df, y=col)

    st.plotly_chart(fig, use_container_width=True)


def _create_scatter_plot(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> None:
    """
    Create and display a scatter plot visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_cols : List[str]
        List of numeric column names available for x and y axes
    categorical_cols : List[str]
        List of categorical column names available for color coding

    Returns
    -------
    None
        This function displays the visualization in the Streamlit UI
    """
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for a scatter plot.")
        return

    col_x = st.selectbox("Select X axis", options=numeric_cols, index=0)
    col_y = st.selectbox("Select Y axis", options=numeric_cols, index=min(1, len(numeric_cols) - 1))

    color_col = None
    if categorical_cols:
        color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)

    if color_col and color_col != "None":
        fig = px.scatter(df, x=col_x, y=col_y, color=color_col)
    else:
        fig = px.scatter(df, x=col_x, y=col_y)

    st.plotly_chart(fig, use_container_width=True)


def _create_bar_chart(df: pd.DataFrame, categorical_cols: list[str]) -> None:
    """
    Create and display a bar chart visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    categorical_cols : List[str]
        List of categorical column names available for the chart

    Returns
    -------
    None
        This function displays the visualization in the Streamlit UI
    """
    if not categorical_cols:
        st.warning("Need at least one categorical column for a bar chart.")
        return

    col_cat = st.selectbox("Select category", options=categorical_cols)

    # Get value counts and create bar chart
    value_counts = df[col_cat].value_counts().reset_index()
    value_counts.columns = [col_cat, "count"]

    fig = px.bar(value_counts, x=col_cat, y="count", title=f"Count of {col_cat}")
    st.plotly_chart(fig, use_container_width=True)
