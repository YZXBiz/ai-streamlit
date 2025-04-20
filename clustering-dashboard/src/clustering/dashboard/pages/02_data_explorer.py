"""Data Exploration Page for Clustering Dashboard.

This page provides comprehensive data exploration tools and summaries.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def display_df_summary(df: pd.DataFrame) -> None:
    """Display a comprehensive summary of the DataFrame with metrics and visualizations.

    Args:
        df: DataFrame to summarize
    """
    # Dataset top-level metrics in visually appealing cards
    st.markdown(
        "<div class='apple-heading'><h3>📋 Dataset Overview</h3></div>", unsafe_allow_html=True
    )
    
    # Key metrics row with 4 cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numeric Columns", len(numeric_cols))
    with col4:
        missing = df.isna().sum().sum()
        st.metric("Missing Values", f"{missing:,}")

    # Data preview and schema in tabs
    st.subheader("🔍 Data Structure")
    data_tabs = st.tabs(["✨ Preview", "📋 Schema", "📊 Data Types"])

    with data_tabs[0]:  # Preview tab
        st.dataframe(df.head(5), use_container_width=True)

    with data_tabs[1]:  # Schema tab
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

    with data_tabs[2]:  # Data Types tab
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

    # Statistical summary in expandable section
    st.subheader("📊 Statistical Summary")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T
        # Add more useful metrics
        stats_df["missing"] = df[numeric_cols].isna().sum()
        stats_df["missing_pct"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)
        
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No numeric columns found in the dataset.")


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Generate a correlation matrix visualization for numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Plotly figure with correlation matrix
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] <= 1:
        return None
        
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmid=0,
        )
    )
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=600,
    )
    
    return fig


def data_explorer_page():
    """Render the data exploration page."""
    st.title("🔍 Data Explorer")
    
    # Check if data is loaded
    if "data" not in st.session_state:
        st.warning("Please upload data in the Data Upload page first.")
        return
    
    df = st.session_state["data"]
    
    # Display data source info
    data_source = st.session_state.get("data_source", "unknown")
    st.info(f"Data loaded from: {data_source.capitalize()}")
    
    # Display dataset summary
    display_df_summary(df)
    
    # Show correlation matrix for numeric features
    st.subheader("Feature Correlation")
    
    corr_fig = plot_correlation_matrix(df)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")
    
    # Column explorer
    st.subheader("Column Explorer")
    
    # Select column to explore
    selected_column = st.selectbox("Select a column to explore:", df.columns)
    
    if selected_column:
        col_data = df[selected_column]
        col_type = df[selected_column].dtype
        
        # Show column statistics
        st.write(f"**Data type:** {col_type}")
        st.write(f"**Unique values:** {col_data.nunique()}")
        st.write(f"**Missing values:** {col_data.isna().sum()} ({col_data.isna().mean()*100:.2f}%)")
        
        # Visualization based on data type
        if np.issubdtype(col_type, np.number):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=col_data, nbinsx=30))
            fig.update_layout(title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show boxplot
            fig = go.Figure()
            fig.add_trace(go.Box(y=col_data))
            fig.update_layout(title=f"Boxplot of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif col_data.nunique() < 20:  # Categorical with few values
            value_counts = col_data.value_counts().reset_index()
            value_counts.columns = [selected_column, "Count"]
            
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=value_counts[selected_column].astype(str),
                    x=value_counts["Count"],
                    orientation="h",
                )
            )
            fig.update_layout(title=f"Value Counts for {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table too
            st.dataframe(value_counts, use_container_width=True)
        
        else:  # Text or high cardinality
            # Show most common values
            st.write("**Most common values:**")
            st.dataframe(
                col_data.value_counts().head(10).reset_index(),
                use_container_width=True
            )


# Run the page
data_explorer_page() 