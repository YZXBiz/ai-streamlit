"""Data summary components for displaying dataset overviews and statistics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from clustering.dashboard.components.metric_card import metric_row


def display_dataset_metrics(df: pd.DataFrame) -> None:
    """Display key metrics about a dataset in a row of metric cards.

    Args:
        df: DataFrame to summarize
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Missing values count
    missing = df.isna().sum().sum()

    # Create metrics list
    metrics = [
        {
            "title": "Total Records",
            "value": f"{df.shape[0]:,}",
            "help_text": "Total number of rows in the dataset",
        },
        {
            "title": "Columns",
            "value": df.shape[1],
            "help_text": "Total number of columns in the dataset",
        },
        {
            "title": "Numeric Columns",
            "value": len(numeric_cols),
            "help_text": "Number of columns with numeric data types",
        },
        {
            "title": "Missing Values",
            "value": f"{missing:,}",
            "help_text": "Total count of missing values in the dataset",
        },
    ]

    # Display metrics
    metric_row(metrics, num_columns=4)


def display_schema_summary(df: pd.DataFrame) -> None:
    """Display a summary of the DataFrame schema.

    Args:
        df: DataFrame to summarize
    """
    buffer = pd.DataFrame(
        {
            "Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Non-Null %": (df.count() / len(df) * 100).round(2).astype(str) + "%",
            "Null Count": df.isna().sum(),
            "Unique Values": [
                df[col].nunique() if df[col].nunique() < 1000 else ">1000" for col in df.columns
            ],
        }
    )
    st.dataframe(buffer, use_container_width=True)


def display_datatype_chart(df: pd.DataFrame) -> None:
    """Create a bar chart showing the count of each data type in the DataFrame.

    Args:
        df: DataFrame to analyze
    """
    # Count columns by data type
    type_counts = df.dtypes.value_counts().reset_index()
    type_counts.columns = ["Data Type", "Count"]

    # Create a horizontal bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=type_counts["Data Type"].astype(str),
            x=type_counts["Count"],
            orientation="h",
            marker_color="#0066CC",
        )
    )

    fig.update_layout(
        title="Column Data Types",
        xaxis_title="Count",
        yaxis_title="Data Type",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_numeric_stats(df: pd.DataFrame) -> None:
    """Display statistical summary of numeric columns.

    Args:
        df: DataFrame to analyze
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        stats_df = df[numeric_cols].describe().T

        # Add more useful metrics
        stats_df["missing"] = df[numeric_cols].isna().sum()
        stats_df["missing_pct"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)

        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No numeric columns found in the dataset.")


def display_complete_data_summary(df: pd.DataFrame) -> None:
    """Display a comprehensive summary of the DataFrame.

    This is a higher-level function that combines multiple summary components.

    Args:
        df: DataFrame to summarize
    """
    # Dataset top-level metrics
    st.markdown(
        "<div class='apple-heading'><h3>📋 Dataset Overview</h3></div>", unsafe_allow_html=True
    )
    display_dataset_metrics(df)

    # Data structure tabs
    st.subheader("🔍 Data Structure")
    data_tabs = st.tabs(["✨ Preview", "📋 Schema", "📊 Data Types"])

    with data_tabs[0]:  # Preview tab
        st.dataframe(df.head(5), use_container_width=True)

    with data_tabs[1]:  # Schema tab
        display_schema_summary(df)

    with data_tabs[2]:  # Data Types tab
        display_datatype_chart(df)

    # Statistical summary
    st.subheader("📊 Statistical Summary")
    display_numeric_stats(df)
