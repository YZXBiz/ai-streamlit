"""Assortment-Clustering Data Explorer Application.

A streamlined application for exploring and visualizing assortment clustering data.
This application allows users to upload data files or connect to Snowflake to explore data visually.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from clustering.dashboard.components.pygwalker_view import get_pyg_renderer
from clustering.io.readers import SnowflakeReader


@st.cache_data
def load_data(file_path: Path | str, file_type: str | None = None) -> pd.DataFrame:
    """Load data from file with caching.

    Args:
        file_path: Path to the data file
        file_type: Type of file (csv, excel, pickle)

    Returns:
        DataFrame loaded from file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Determine file type from extension if not provided
    if file_type is None:
        suffix = file_path.suffix.lower()
        if suffix in (".csv", ".tsv", ".txt"):
            file_type = "csv"
        elif suffix in (".xls", ".xlsx"):
            file_type = "excel"
        elif suffix == ".pkl":
            file_type = "pickle"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # Load data based on file type
    try:
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "excel":
            return pd.read_excel(file_path)
        elif file_type == "pickle":
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def load_snowflake_data(query: str, use_cache: bool = True) -> pd.DataFrame:
    """Load data from Snowflake with caching.

    Args:
        query: SQL query to execute against Snowflake
        use_cache: Whether to use local caching for the query

    Returns:
        DataFrame loaded from Snowflake
    """
    try:
        # Use the SnowflakeReader from clustering.io
        reader = SnowflakeReader(query=query, use_cache=use_cache)

        # Read data using the reader
        df = reader.read()

        # Convert from polars to pandas DataFrame
        return df.to_pandas()
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {str(e)}")
        return pd.DataFrame()


def display_df_summary(df: pd.DataFrame) -> None:
    """Display a comprehensive summary of the DataFrame with professional metrics and visualizations.

    Args:
        df: DataFrame to summarize
    """
    # Dataset top-level metrics in visually appealing cards
    st.markdown("<h3>📋 Dataset Overview</h3>", unsafe_allow_html=True)

    # Key metrics row with 4 cards
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{df.shape[0]:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Numeric Columns</div>
                <div class="metric-value">{len(numeric_cols)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        missing = df.isna().sum().sum()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{missing:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data preview and schema in tabs
    st.markdown("<h3>🔍 Data Structure</h3>", unsafe_allow_html=True)
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

    # Statistical summary in expandable section
    st.markdown("<h3>📊 Statistical Summary</h3>", unsafe_allow_html=True)
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)

    with st.expander("View Numerical Statistics", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_df = df[numeric_cols].describe().T
            # Add more useful metrics
            stats_df["missing"] = df[numeric_cols].isna().sum().values
            stats_df["missing_pct"] = (df[numeric_cols].isna().sum() / len(df) * 100).values.round(
                2
            )
            stats_df["zeros"] = (df[numeric_cols] == 0).sum().values
            stats_df["zeros_pct"] = ((df[numeric_cols] == 0).sum() / len(df) * 100).values.round(2)

            # Format index to be more readable
            stats_df.index.name = "Column"
            stats_df = stats_df.reset_index()
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numeric columns found in this dataset")

    st.markdown("</div>", unsafe_allow_html=True)

    # Sample of selected columns (if too many columns)
    if len(df.columns) > 8:
        st.markdown("<h3>📊 Column Highlights</h3>", unsafe_allow_html=True)

        # Select a sample of diverse columns
        numeric_sample = numeric_cols[: min(3, len(numeric_cols))]
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_sample = categorical_cols[: min(3, len(categorical_cols))]

        sample_cols = numeric_sample + categorical_sample

        if sample_cols:
            for i in range(0, len(sample_cols), 3):
                cols = st.columns(min(3, len(sample_cols) - i))
                for j, col in enumerate(cols):
                    if i + j < len(sample_cols):
                        column_name = sample_cols[i + j]
                        with col:
                            st.markdown(f"**{column_name}**")
                            if column_name in numeric_cols:
                                # For numeric columns, show a small histogram
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Histogram(x=df[column_name].dropna(), marker_color="#0066CC")
                                )
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, b=0, t=0),
                                    template="plotly_white",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # For categorical columns, show top values
                                value_counts = df[column_name].value_counts().head(5)
                                st.write(
                                    f"Top values ({min(5, len(value_counts))} of {df[column_name].nunique()}):"
                                )
                                st.write(value_counts)


def create_metric_card(title: str, value: Any, prefix: str = "", suffix: str = "") -> None:
    """Create a styled metric card.

    Args:
        title: Title of the metric
        value: Value to display
        prefix: Prefix to display before the value (e.g., "$")
        suffix: Suffix to display after the value (e.g., "%")
    """
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a distribution plot for a numerical column.

    Args:
        df: DataFrame containing the data
        column: Name of the numerical column to plot

    Returns:
        Plotly figure object with the distribution plot
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution", "Box Plot"),
        specs=[[{"type": "histogram"}, {"type": "box"}]],
    )

    # Histogram
    fig.add_trace(go.Histogram(x=df[column], name=column, marker_color="#0066CC"), row=1, col=1)

    # Box plot
    fig.add_trace(go.Box(y=df[column], name=column, marker_color="#0066CC"), row=1, col=2)

    fig.update_layout(
        title=f"Distribution Analysis: {column}",
        height=400,
        showlegend=False,
        template="plotly_white",
    )

    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Create a correlation matrix heatmap.

    Args:
        df: DataFrame containing numerical data

    Returns:
        Plotly figure object with the correlation heatmap
    """
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues", zmin=-1, zmax=1
        )
    )

    fig.update_layout(title="Correlation Matrix", height=600, width=700, template="plotly_white")

    return fig


def add_custom_css():
    """Add custom CSS for styling the dashboard with an Apple-inspired design."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .content-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .content-container:hover {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }
        
        /* Base styles */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 500;
        }
        
        h1 {
            color: #1d1d1f;
            font-size: 2.3rem !important;
            letter-spacing: -0.02em;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e8e8ed;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            color: #1d1d1f;
            font-size: 1.6rem !important;
            letter-spacing: -0.01em;
            padding-top: 1rem;
        }
        
        h3 {
            color: #1d1d1f;
            font-size: 1.3rem !important;
            letter-spacing: -0.01em;
        }

        p, li, div {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1d1d1f;
        }
        
        /* Apple-inspired tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #f0f0f0;
            padding-bottom: 8px;
        }
        
        .stTabs [role="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            background-color: #f5f5f7;
            border: none !important;
            color: #1d1d1f;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stTabs [role="tab"]:hover {
            background-color: #0071e3;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #0071e3;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Fix white space below header */
        .stTabs [data-baseweb="tab-panel"] {
            padding: 0 !important;
        }
        
        .stTabs {
            margin-bottom: 0 !important;
        }
        
        .main .block-container {
            padding-top: 0;
        }
        
        /* Additional fix for Dataset Overview tab */
        div[data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
            background-color: #f5f5f7;
            color: #1d1d1f;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.07);
        }
        
        .stButton button:hover {
            background-color: #e8e8ed;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Card styling for metrics */
        .metric-card {
            padding: 10px 15px;
            border-radius: 10px;
            background: linear-gradient(to bottom right, #f8f8f8, #ffffff);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #6e6e73;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        /* Data table styling */
        div.stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e8e8ed;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        div.stDataFrame td {
            text-align: left;
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
        }
        
        /* Card style for sections */
        .card {
            border-radius: 12px;
            background-color: #ffffff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e8e8ed;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            border-radius: 8px;
            background-color: #f5f5f7 !important;
            color: #1d1d1f !important;
            font-weight: 500;
        }
        
        .streamlit-expanderContent {
            border-radius: 0 0 8px 8px;
            border: 1px solid #f0f0f0;
            border-top: none;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #f5f5f7;
            border-right: 1px solid #e8e8ed;
        }
        
        [data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio > div {
            padding: 0.3rem 0;
        }
        
        .stRadio label, .stCheckbox label {
            font-size: 0.95rem;
            color: #1d1d1f;
        }
        
        /* Section dividers and containers */
        .section-divider {
            margin: 2.5rem 0 1.5rem 0;
            height: 1px;
            background-color: #e8e8ed;
        }
        
        .section-header {
            font-size: 1.3rem;
            font-weight: 500;
            color: #1d1d1f;
            padding-bottom: 0.8rem;
            margin-bottom: 1.2rem;
            letter-spacing: -0.01em;
        }
        
        .section-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #e8e8ed;
        }
        
        h4.subsection-header {
            margin-top: 1.5rem;
            color: #1d1d1f;
            font-size: 1.1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e8e8ed;
            font-weight: 500;
            letter-spacing: -0.01em;
        }
        
        /* Legacy metric styling for compatibility */
        .apple-metric {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            padding: 1.2rem;
            flex: 1;
            min-width: 180px;
            transition: all 0.2s ease;
            border: 1px solid rgba(0, 0, 0, 0.06);
        }
        
        .apple-metric:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        }
        
        .apple-metric h3 {
            margin: 0;
            color: #1d1d1f;
            font-size: 2rem;
            font-weight: 600;
            line-height: 1.1;
        }
        
        .apple-metric p {
            margin: 0.5rem 0 0 0;
            color: #86868b;
            font-size: 0.95rem;
            font-weight: 400;
        }
        
        /* Nicer dividers */
        hr {
            height: 1px;
            background-color: #e6e6e6;
            border: none;
            margin: 20px 0;
        }
        
        /* Refinements to selectbox and inputs */
        .stSelectbox, .stMultiSelect {
            margin-bottom: 15px;
        }
        
        [data-baseweb="select"] {
            border-radius: 8px;
        }
        
        /* Hide the default Streamlit menus and padding */
        #MainMenu, footer, header {
            visibility: hidden;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        
        /* Snowflake mode styling */
        body.snowflake-mode .navigation-section {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Run the Assortment-Clustering Explorer application."""
    # Set page config
    st.set_page_config(
        page_title="Assortment-Clustering Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add custom CSS for Apple-inspired styling
    add_custom_css()

    # Initialize session state variables if they don't exist
    if "data" not in st.session_state:
        st.session_state.data = None

    if "data_source" not in st.session_state:
        st.session_state.data_source = "📄 File Upload"

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Explorer"

    # Header with logo and title in a modern layout
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(
            """
        <div style='text-align: center; padding: 10px;'>
            <span style='font-size: 3rem; color: #4F46E5;'>📊</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <h1 style='margin-bottom: 0;'>Assortment-Clustering Explorer</h1>
        <p style='color: #6B7280; margin-top: 0;'>
            Interactive data visualization for cluster analysis and exploration
        </p>
        """,
            unsafe_allow_html=True,
        )

    # Sidebar for data source options only
    with st.sidebar:
        st.markdown(
            """
        <h2 style='color: #000000; margin-bottom: 1.5rem; border-bottom: 1px solid #818CF8; padding-bottom: 0.5rem; font-weight: 600; font-size: 1.25rem;'>
            <span style="margin-right: 8px;">📂</span> Data Source
        </h2>
        """,
            unsafe_allow_html=True,
        )

        # Initialize variables to avoid 'before assignment' errors
        uploaded_file = None
        snowflake_query = ""
        load_sf_data = False
        use_cache = True

        # Source selection tabs with icons
        data_source = st.radio(
            "Select Data Source",
            ["📄 File Upload", "❄️ Snowflake"],
            index=0 if st.session_state.data_source == "📄 File Upload" else 1,
            format_func=lambda x: x,
        )

        # Update data source in session state
        st.session_state.data_source = data_source

        # Add special class to body when in Snowflake mode to hide navigation
        if "❄️ Snowflake" in data_source:
            st.markdown(
                """
            <script>
                document.body.classList.add('snowflake-mode');
            </script>
            """,
                unsafe_allow_html=True,
            )

        # Store current data source for comparison on next render
        st.session_state.last_data_source = data_source

        if "📄 File Upload" in data_source:
            # File upload with enhanced UI
            st.markdown("### Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "xls", "pkl"],
                help="Upload a CSV, Excel, or Pickle file",
                key="data_upload",
            )

            if uploaded_file is None:
                st.info("👆 Drag and drop your file here or click to browse")

        elif "❄️ Snowflake" in data_source:
            # Snowflake query input with cleaner UI
            st.markdown("### Query Snowflake Database")

            # Pre-populate the query if it exists in session state
            query_value = ""
            if "snowflake_query" in st.session_state:
                query_value = st.session_state.snowflake_query

            st.info("Enter SQL to retrieve data from Snowflake")
            snowflake_query = st.text_area(
                "SQL Query",
                value=query_value,
                height=150,
                placeholder="SELECT * FROM your_table LIMIT 1000",
            )

            # Save query to session state
            if snowflake_query != query_value:
                st.session_state.snowflake_query = snowflake_query

            use_cache = st.checkbox("💾 Use cached results (if available)", value=True)

            # More prominent load button
            load_sf_data = st.button(
                "🚀 Load Data from Snowflake", type="primary", use_container_width=True
            )

    # Add a loading animation for better UX
    with st.spinner("Processing data..."):
        # Load data based on selected source
        if "📄 File Upload" in data_source and uploaded_file is not None:
            # Save uploaded file to temp location
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the data
            st.session_state.data = load_data(temp_path)

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Show success message
            if not st.session_state.data.empty:
                st.success(
                    f"✅ Successfully loaded {len(st.session_state.data):,} rows from {uploaded_file.name}"
                )

        elif "❄️ Snowflake" in data_source and load_sf_data and snowflake_query:
            # Load data from Snowflake with the provided query
            with st.spinner("Connecting to Snowflake..."):
                st.session_state.data = load_snowflake_data(
                    query=snowflake_query, use_cache=use_cache
                )

                if not st.session_state.data.empty:
                    st.success(
                        f"✅ Successfully loaded {len(st.session_state.data):,} rows from Snowflake"
                    )
                else:
                    st.error("❌ No data returned from Snowflake query")

    # Display visualization if data is loaded
    if st.session_state.data is not None:
        # Create tabs for Dataset Overview and Interactive Visualization with Apple-style design
        tab1, tab2 = st.tabs(["Dataset Overview", "Interactive Visualization"])

        # Dataset Overview tab
        with tab1:
            st.markdown(
                """
                <div style="background-color: #f5f5f7; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0066CC; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
                    <h2 style="margin-top: 0; color: #1d1d1f; font-size: 1.6rem; letter-spacing: -0.01em;">Dataset Overview</h2>
                    <p style="color: #86868b; margin-bottom: 0; font-size: 1rem;">
                        Complete analysis of your dataset's structure, statistics, and quality metrics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Add CSS for section styling - Apple inspired
            st.markdown(
                """
            <style>
                .section-divider {
                    margin: 2.5rem 0 1.5rem 0;
                    height: 1px;
                    background-color: #e8e8ed;
                }
                .section-header {
                    font-size: 1.3rem;
                    font-weight: 500;
                    color: #1d1d1f;
                    letter-spacing: -0.01em;
                    padding-bottom: 0.8rem;
                    margin-bottom: 1.5rem;
                }
                .section-container {
                    background-color: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    margin-bottom: 2rem;
                    border: 1px solid #e8e8ed;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Use the enhanced display_df_summary function
            display_df_summary(st.session_state.data)

            # Add immediate interactive EDA visualizations - with better visual separation
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h3 class='section-header'>🔍 Interactive Data Explorer</h3>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            # Quick column exploration
            cols = st.multiselect(
                "Select columns to explore",
                st.session_state.data.columns.tolist(),
                default=st.session_state.data.columns.tolist()[:3]
                if len(st.session_state.data.columns) > 3
                else st.session_state.data.columns.tolist(),
            )

            if cols:
                st.dataframe(st.session_state.data[cols].head(10), use_container_width=True)

                # Get column types for selected columns
                numeric_cols = [
                    col for col in cols if pd.api.types.is_numeric_dtype(st.session_state.data[col])
                ]
                categorical_cols = [col for col in cols if col not in numeric_cols]

                # Show immediate visualizations for selected columns
                if numeric_cols:
                    st.markdown(
                        "<h4 style='margin-top:1.5rem; color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Quick Numeric Column Analysis</h4>",
                        unsafe_allow_html=True,
                    )

                    # Display histogram or boxplot for first numeric column
                    if numeric_cols:
                        viz_cols = st.columns(min(3, len(numeric_cols)))
                        for i, col_container in enumerate(viz_cols):
                            if i < len(numeric_cols):
                                with col_container:
                                    col_name = numeric_cols[i]
                                    st.markdown(
                                        f"<p style='font-weight:600; font-size:1.1rem;'>{col_name}</p>",
                                        unsafe_allow_html=True,
                                    )

                                    # Simple histogram
                                    fig = go.Figure()
                                    fig.add_trace(
                                        go.Histogram(
                                            x=st.session_state.data[col_name].dropna(),
                                            marker_color="#0066CC",
                                        )
                                    )
                                    fig.update_layout(
                                        height=200,
                                        margin=dict(l=10, r=10, b=30, t=10),
                                        xaxis_title=col_name,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show quick stats
                                    st.metric(
                                        "Mean", f"{st.session_state.data[col_name].mean():.2f}"
                                    )
                                    min_val = st.session_state.data[col_name].min()
                                    max_val = st.session_state.data[col_name].max()
                                    st.write(f"Range: {min_val:.2f} - {max_val:.2f}")

                # Show categorical data visualizations
                if categorical_cols:
                    st.markdown(
                        "<h4 style='margin-top:1.5rem; color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Quick Categorical Column Analysis</h4>",
                        unsafe_allow_html=True,
                    )
                    cat_viz_cols = st.columns(min(2, len(categorical_cols)))

                    for i, col_container in enumerate(cat_viz_cols):
                        if i < len(categorical_cols):
                            with col_container:
                                col_name = categorical_cols[i]
                                st.markdown(
                                    f"<p style='font-weight:600; font-size:1.1rem;'>{col_name}</p>",
                                    unsafe_allow_html=True,
                                )

                                # Get value counts
                                value_counts = (
                                    st.session_state.data[col_name].value_counts().head(10)
                                )

                                # Create bar chart
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Bar(
                                        x=value_counts.index,
                                        y=value_counts.values,
                                        marker_color="#0066CC",
                                    )
                                )
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=10, r=10, b=30, t=10),
                                    xaxis_title=col_name,
                                    yaxis_title="Count",
                                )
                                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Advanced data quality section
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h3 class='section-header'>🔍 Data Quality Assessment</h3>", unsafe_allow_html=True
            )

            quality_tabs = st.tabs(["Completeness", "Consistency", "Distribution Analysis"])

            with quality_tabs[0]:  # Completeness tab
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                # Calculate completeness metrics
                total_cells = st.session_state.data.shape[0] * st.session_state.data.shape[1]
                missing_cells = st.session_state.data.isna().sum().sum()
                completeness_pct = 100 - (missing_cells / total_cells * 100)

                completeness_col1, completeness_col2 = st.columns([1, 2])
                with completeness_col1:
                    st.markdown(
                        f"""
                    <div class="apple-metric">
                        <h3>{completeness_pct:.2f}%</h3>
                        <p>Overall Completeness</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Missing values by column
                    st.markdown(
                        "<h4 style='margin-top:1.5rem; color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Missing Values by Column</h4>",
                        unsafe_allow_html=True,
                    )
                    missing_by_col = st.session_state.data.isna().sum().sort_values(ascending=False)
                    missing_by_col_pct = (
                        missing_by_col / st.session_state.data.shape[0] * 100
                    ).round(2)
                    missing_df = pd.DataFrame(
                        {"Missing Count": missing_by_col, "Missing %": missing_by_col_pct}
                    ).reset_index()
                    missing_df.columns = ["Column", "Missing Count", "Missing %"]
                    st.dataframe(missing_df.head(10), use_container_width=True)

                with completeness_col2:
                    # Create a bar chart for missing values
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=missing_by_col.index[:15],  # Show top 15 columns with missing values
                            y=missing_by_col_pct[:15],
                            marker_color="#0066CC",
                        )
                    )
                    fig.update_layout(
                        title="Top Columns with Missing Values (%)",
                        xaxis_title="Column",
                        yaxis_title="Missing (%)",
                        height=400,
                        margin=dict(l=20, r=20, b=30, t=40),
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with quality_tabs[1]:  # Consistency tab
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                st.markdown(
                    "<h4 style='color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Value Consistency</h4>",
                    unsafe_allow_html=True,
                )

                # Select numeric columns
                numeric_cols = st.session_state.data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                consistency_col1, consistency_col2 = st.columns([1, 1])

                with consistency_col1:
                    if len(numeric_cols) > 0:
                        selected_col = st.selectbox(
                            "Select column for consistency check", numeric_cols
                        )
                        # Calculate potential duplicates and outliers
                        unique_values = st.session_state.data[selected_col].nunique()
                        duplicate_rows = st.session_state.data.duplicated().sum()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f"""
                            <div class="apple-metric">
                                <h3>{unique_values}</h3>
                                <p>Unique Values</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(
                                f"""
                            <div class="apple-metric">
                                <h3>{duplicate_rows}</h3>
                                <p>Duplicate Rows</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        # Create boxplot to identify outliers
                        fig = go.Figure()
                        fig.add_trace(
                            go.Box(
                                y=st.session_state.data[selected_col].dropna(), name=selected_col
                            )
                        )
                        fig.update_layout(
                            title=f"Distribution and Outliers: {selected_col}", height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric columns available for consistency analysis")

                with consistency_col2:
                    categorical_cols = st.session_state.data.select_dtypes(
                        include=["object"]
                    ).columns.tolist()
                    if len(categorical_cols) > 0:
                        selected_cat_col = st.selectbox(
                            "Select categorical column", categorical_cols
                        )
                        value_counts = (
                            st.session_state.data[selected_cat_col].value_counts().reset_index()
                        )
                        value_counts.columns = ["Value", "Count"]

                        # Create pie chart for categorical distribution
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=value_counts["Value"][:10],  # Top 10 values
                                    values=value_counts["Count"][:10],
                                    hole=0.3,
                                    marker_colors=px.colors.sequential.Blues_r,
                                )
                            ]
                        )
                        fig.update_layout(title=f"Top 10 Values: {selected_cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categorical columns available for consistency analysis")
                st.markdown("</div>", unsafe_allow_html=True)

            with quality_tabs[2]:  # Distribution Analysis
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                st.markdown(
                    "<h4 style='color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Distribution Analysis</h4>",
                    unsafe_allow_html=True,
                )

                dist_col1, dist_col2 = st.columns([1, 1])

                with dist_col1:
                    if len(numeric_cols) > 0:
                        selected_dist_col = st.selectbox(
                            "Select column for distribution", numeric_cols
                        )

                        # Create histogram
                        fig = go.Figure()
                        fig.add_trace(
                            go.Histogram(
                                x=st.session_state.data[selected_dist_col].dropna(),
                                marker_color="#0066CC",
                                nbinsx=30,
                            )
                        )
                        fig.update_layout(
                            title=f"Histogram: {selected_dist_col}",
                            xaxis_title=selected_dist_col,
                            yaxis_title="Frequency",
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Add descriptive statistics
                        desc_stats = (
                            st.session_state.data[selected_dist_col].describe().reset_index()
                        )
                        desc_stats.columns = ["Statistic", "Value"]
                        st.dataframe(desc_stats, use_container_width=True)
                    else:
                        st.info("No numeric columns available for distribution analysis")

                with dist_col2:
                    if len(numeric_cols) > 1:
                        x_col = st.selectbox("X-axis", numeric_cols, index=0)
                        y_col = st.selectbox(
                            "Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1)
                        )

                        # Create scatter plot
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=st.session_state.data[x_col],
                                y=st.session_state.data[y_col],
                                mode="markers",
                                marker=dict(color="#0066CC", size=5, opacity=0.6),
                            )
                        )
                        fig.update_layout(
                            title=f"Relationship: {x_col} vs {y_col}",
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate correlation
                        correlation = st.session_state.data[[x_col, y_col]].corr().iloc[0, 1]
                        st.metric("Correlation", f"{correlation:.4f}")
                    else:
                        st.info("Need at least two numeric columns for relationship analysis")
                st.markdown("</div>", unsafe_allow_html=True)

            # Add interactive correlation heatmap (removed PCA and Time Series)
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h3 class='section-header'>🧪 Feature Correlations</h3>", unsafe_allow_html=True
            )

            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                selected_features = st.multiselect(
                    "Select features to analyze correlations",
                    numeric_cols,
                    default=numeric_cols[: min(8, len(numeric_cols))],
                )

                if selected_features and len(selected_features) > 1:
                    # Create correlation heatmap
                    corr_matrix = st.session_state.data[selected_features].corr()

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale="Blues",
                            zmin=-1,
                            zmax=1,
                        )
                    )
                    fig.update_layout(
                        title="Feature Correlation Matrix",
                        height=500,
                        margin=dict(l=40, r=40, b=40, t=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Strong correlations
                    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                    corr_pairs = corr_pairs[corr_pairs < 1.0]  # Remove self-correlations
                    strong_corr = corr_pairs[abs(corr_pairs) > 0.5].sort_values(ascending=False)

                    if not strong_corr.empty:
                        st.markdown(
                            "<h4 style='color:#2563EB; font-size:1.2rem; padding-bottom:0.5rem; border-bottom:1px solid #e5e7eb;'>Strong Correlations</h4>",
                            unsafe_allow_html=True,
                        )
                        strong_corr_df = pd.DataFrame(
                            {
                                "Feature Pair": [
                                    f"{idx[0]} — {idx[1]}" for idx in strong_corr.index
                                ],
                                "Correlation": strong_corr.values,
                            }
                        )
                        st.dataframe(strong_corr_df, use_container_width=True)
                    else:
                        st.info("No strong correlations found between selected features")
            else:
                st.info("Need at least two numeric columns for correlation analysis")
            st.markdown("</div>", unsafe_allow_html=True)

            # Remove the entire Data Explorer Widget section from here
            # It's already available in the Interactive Visualization tab

        # Interactive Visualization tab
        with tab2:
            st.markdown("## Interactive Visualization")

            # Get cached renderer
            renderer = get_pyg_renderer(st.session_state.data)

            if renderer:
                # Render selected view
                if st.session_state.view_mode == "Explorer":
                    # Use the explorer method for the standard view
                    renderer.explorer(key="viz_tab_explorer")
                elif st.session_state.view_mode == "Chart":
                    # For Chart view, we need to specify an index for which chart to display
                    try:
                        # Show the first chart (index 0)
                        renderer.chart(0, key="viz_tab_chart")
                    except Exception as e:
                        st.warning("⚠️ No charts available. Create a chart in Explorer view first.")
                        st.error(f"Error: {str(e)}")
                elif st.session_state.view_mode == "Data Profiling":
                    # For Data Profiling, we use explorer with "data" as the default tab
                    renderer.explorer(default_tab="data", key="viz_tab_profiling")
                elif st.session_state.view_mode == "Data Preview":
                    st.dataframe(st.session_state.data, use_container_width=True)
    else:
        # Enhanced empty state with illustrations and guidance
        if "📄 File Upload" in data_source:
            st.markdown(
                """
            <div style='text-align: center; padding: 3rem 1rem; background-color: white; border-radius: 0.5rem; margin: 2rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1.5rem;'>📤</div>
                <h3>Upload Your Dataset to Begin</h3>
                <p style='color: #6B7280; max-width: 600px; margin: 1rem auto;'>
                    Select a CSV, Excel, or Pickle file from your device to begin exploring your cluster data.
                    Use the file uploader in the sidebar to get started.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif "❄️ Snowflake" in data_source:
            st.markdown(
                """
            <div style='text-align: center; padding: 3rem 1rem; background-color: white; border-radius: 0.5rem; margin: 2rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1.5rem;'>❄️</div>
                <h3>Connect to Snowflake</h3>
                <p style='color: #6B7280; max-width: 600px; margin: 1rem auto;'>
                    Enter your SQL query in the sidebar and click "Load Data from Snowflake" to begin exploring your data.
                    For best performance, add limits to your query.
                </p>
                <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 0.3rem; margin-top: 1.5rem; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto;'>
                    <code>SELECT * FROM your_table WHERE cluster_id IS NOT NULL LIMIT 1000</code>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Enhanced footer with version info and credits
    st.sidebar.markdown("<hr style='margin: 2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='color: #94A3B8; margin-bottom: 0.5rem;'>Assortment-Clustering Explorer v1.0</p>
        <p style='color: #94A3B8; font-size: 0.8rem; margin-top: 0;'>
            Created by Jackson with ❤️
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
